//===----- UninitializedObjectChecker.cpp ------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a checker that reports uninitialized fields in objects
// created after a constructor call.
//
// This checker has several options:
//   - "Pedantic" (boolean). If its not set or is set to false, the checker
//     won't emit warnings for objects that don't have at least one initialized
//     field. This may be set with
//
//     `-analyzer-config alpha.cplusplus.UninitializedObject:Pedantic=true`.
//
//   - "NotesAsWarnings" (boolean). If set to true, the checker will emit a
//     warning for each uninitalized field, as opposed to emitting one warning
//     per constructor call, and listing the uninitialized fields that belongs
//     to it in notes. Defaults to false.
//
//     `-analyzer-config \
//         alpha.cplusplus.UninitializedObject:NotesAsWarnings=true`.
//
//   - "CheckPointeeInitialization" (boolean). If set to false, the checker will
//     not analyze the pointee of pointer/reference fields, and will only check
//     whether the object itself is initialized. Defaults to false.
//
//     `-analyzer-config \
//         alpha.cplusplus.UninitializedObject:CheckPointeeInitialization=true`.
//
//     TODO: With some clever heuristics, some pointers should be dereferenced
//     by default. For example, if the pointee is constructed within the
//     constructor call, it's reasonable to say that no external object
//     references it, and we wouldn't generate multiple report on the same
//     pointee.
//
// To read about how the checker works, refer to the comments in
// UninitializedObject.h.
//
// Some of the logic is implemented in UninitializedPointee.cpp, to reduce the
// complexity of this file.
//
//===----------------------------------------------------------------------===//

#include "UninitializedObject.h"
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeMap.h"

using namespace clang;
using namespace clang::ento;

namespace {

class UninitializedObjectChecker : public Checker<check::EndFunction> {
  std::unique_ptr<BuiltinBug> BT_uninitField;

public:
  // These fields will be initialized when registering the checker.
  bool IsPedantic;
  bool ShouldConvertNotesToWarnings;
  bool CheckPointeeInitialization;

  UninitializedObjectChecker()
      : BT_uninitField(new BuiltinBug(this, "Uninitialized fields")) {}
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
};

} // end of anonymous namespace

// Utility function declarations.

/// Returns the object that was constructed by CtorDecl, or None if that isn't
/// possible.
// TODO: Refactor this function so that it returns the constructed object's
// region.
static Optional<nonloc::LazyCompoundVal>
getObjectVal(const CXXConstructorDecl *CtorDecl, CheckerContext &Context);

/// Checks whether the object constructed by \p Ctor will be analyzed later
/// (e.g. if the object is a field of another object, in which case we'd check
/// it multiple times).
static bool willObjectBeAnalyzedLater(const CXXConstructorDecl *Ctor,
                               CheckerContext &Context);

/// Constructs a note message for a given FieldChainInfo object.
static void printNoteMessage(llvm::raw_ostream &Out,
                             const FieldChainInfo &Chain);

/// Returns with Field's name. This is a helper function to get the correct name
/// even if Field is a captured lambda variable.
static StringRef getVariableName(const FieldDecl *Field);

//===----------------------------------------------------------------------===//
//                  Methods for UninitializedObjectChecker.
//===----------------------------------------------------------------------===//

void UninitializedObjectChecker::checkEndFunction(
    const ReturnStmt *RS, CheckerContext &Context) const {

  const auto *CtorDecl = dyn_cast_or_null<CXXConstructorDecl>(
      Context.getLocationContext()->getDecl());
  if (!CtorDecl)
    return;

  if (!CtorDecl->isUserProvided())
    return;

  if (CtorDecl->getParent()->isUnion())
    return;

  // This avoids essentially the same error being reported multiple times.
  if (willObjectBeAnalyzedLater(CtorDecl, Context))
    return;

  Optional<nonloc::LazyCompoundVal> Object = getObjectVal(CtorDecl, Context);
  if (!Object)
    return;

  FindUninitializedFields F(Context.getState(), Object->getRegion(), IsPedantic,
                            CheckPointeeInitialization);

  const UninitFieldSet &UninitFields = F.getUninitFields();

  if (UninitFields.empty())
    return;

  // There are uninitialized fields in the record.

  ExplodedNode *Node = Context.generateNonFatalErrorNode(Context.getState());
  if (!Node)
    return;

  PathDiagnosticLocation LocUsedForUniqueing;
  const Stmt *CallSite = Context.getStackFrame()->getCallSite();
  if (CallSite)
    LocUsedForUniqueing = PathDiagnosticLocation::createBegin(
        CallSite, Context.getSourceManager(), Node->getLocationContext());

  // For Plist consumers that don't support notes just yet, we'll convert notes
  // to warnings.
  if (ShouldConvertNotesToWarnings) {
    for (const auto &Chain : UninitFields) {
      SmallString<100> WarningBuf;
      llvm::raw_svector_ostream WarningOS(WarningBuf);

      printNoteMessage(WarningOS, Chain);

      auto Report = llvm::make_unique<BugReport>(
          *BT_uninitField, WarningOS.str(), Node, LocUsedForUniqueing,
          Node->getLocationContext()->getDecl());
      Context.emitReport(std::move(Report));
    }
    return;
  }

  SmallString<100> WarningBuf;
  llvm::raw_svector_ostream WarningOS(WarningBuf);
  WarningOS << UninitFields.size() << " uninitialized field"
            << (UninitFields.size() == 1 ? "" : "s")
            << " at the end of the constructor call";

  auto Report = llvm::make_unique<BugReport>(
      *BT_uninitField, WarningOS.str(), Node, LocUsedForUniqueing,
      Node->getLocationContext()->getDecl());

  for (const auto &Chain : UninitFields) {
    SmallString<200> NoteBuf;
    llvm::raw_svector_ostream NoteOS(NoteBuf);

    printNoteMessage(NoteOS, Chain);

    Report->addNote(NoteOS.str(),
                    PathDiagnosticLocation::create(Chain.getEndOfChain(),
                                                   Context.getSourceManager()));
  }
  Context.emitReport(std::move(Report));
}

//===----------------------------------------------------------------------===//
//                   Methods for FindUninitializedFields.
//===----------------------------------------------------------------------===//

FindUninitializedFields::FindUninitializedFields(
    ProgramStateRef State, const TypedValueRegion *const R, bool IsPedantic,
    bool CheckPointeeInitialization)
    : State(State), ObjectR(R), IsPedantic(IsPedantic),
      CheckPointeeInitialization(CheckPointeeInitialization) {}

const UninitFieldSet &FindUninitializedFields::getUninitFields() {
  isNonUnionUninit(ObjectR, FieldChainInfo(Factory));

  if (!IsPedantic && !IsAnyFieldInitialized)
    UninitFields.clear();

  return UninitFields;
}

bool FindUninitializedFields::addFieldToUninits(FieldChainInfo Chain) {
  if (State->getStateManager().getContext().getSourceManager().isInSystemHeader(
          Chain.getEndOfChain()->getLocation()))
    return false;

  return UninitFields.insert(Chain).second;
}

bool FindUninitializedFields::isNonUnionUninit(const TypedValueRegion *R,
                                               FieldChainInfo LocalChain) {
  assert(R->getValueType()->isRecordType() &&
         !R->getValueType()->isUnionType() &&
         "This method only checks non-union record objects!");

  const RecordDecl *RD =
      R->getValueType()->getAs<RecordType>()->getDecl()->getDefinition();
  assert(RD && "Referred record has no definition");

  bool ContainsUninitField = false;

  // Are all of this non-union's fields initialized?
  for (const FieldDecl *I : RD->fields()) {

    const auto FieldVal =
        State->getLValue(I, loc::MemRegionVal(R)).castAs<loc::MemRegionVal>();
    const auto *FR = FieldVal.getRegionAs<FieldRegion>();
    QualType T = I->getType();

    // If LocalChain already contains FR, then we encountered a cyclic
    // reference. In this case, region FR is already under checking at an
    // earlier node in the directed tree.
    if (LocalChain.contains(FR))
      return false;

    if (T->isStructureOrClassType()) {
      if (isNonUnionUninit(FR, {LocalChain, FR}))
        ContainsUninitField = true;
      continue;
    }

    if (T->isUnionType()) {
      if (isUnionUninit(FR)) {
        if (addFieldToUninits({LocalChain, FR}))
          ContainsUninitField = true;
      } else
        IsAnyFieldInitialized = true;
      continue;
    }

    if (T->isArrayType()) {
      IsAnyFieldInitialized = true;
      continue;
    }

    if (T->isPointerType() || T->isReferenceType() || T->isBlockPointerType()) {
      if (isPointerOrReferenceUninit(FR, LocalChain))
        ContainsUninitField = true;
      continue;
    }

    if (isPrimitiveType(T)) {
      SVal V = State->getSVal(FieldVal);

      if (isPrimitiveUninit(V)) {
        if (addFieldToUninits({LocalChain, FR}))
          ContainsUninitField = true;
      }
      continue;
    }

    llvm_unreachable("All cases are handled!");
  }

  // Checking bases.
  // FIXME: As of now, because of `willObjectBeAnalyzedLater`, objects whose
  // type is a descendant of another type will emit warnings for uninitalized
  // inherited members.
  // This is not the only way to analyze bases of an object -- if we didn't
  // filter them out, and didn't analyze the bases, this checker would run for
  // each base of the object in order of base initailization and in theory would
  // find every uninitalized field. This approach could also make handling
  // diamond inheritances more easily.
  //
  // This rule (that a descendant type's cunstructor is responsible for
  // initializing inherited data members) is not obvious, and should it should
  // be.
  const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD);
  if (!CXXRD)
    return ContainsUninitField;

  for (const CXXBaseSpecifier &BaseSpec : CXXRD->bases()) {
    const auto *BaseRegion = State->getLValue(BaseSpec, R)
                                 .castAs<loc::MemRegionVal>()
                                 .getRegionAs<TypedValueRegion>();

    if (isNonUnionUninit(BaseRegion, LocalChain))
      ContainsUninitField = true;
  }

  return ContainsUninitField;
}

bool FindUninitializedFields::isUnionUninit(const TypedValueRegion *R) {
  assert(R->getValueType()->isUnionType() &&
         "This method only checks union objects!");
  // TODO: Implement support for union fields.
  return false;
}

bool FindUninitializedFields::isPrimitiveUninit(const SVal &V) {
  if (V.isUndef())
    return true;

  IsAnyFieldInitialized = true;
  return false;
}

//===----------------------------------------------------------------------===//
//                       Methods for FieldChainInfo.
//===----------------------------------------------------------------------===//

FieldChainInfo::FieldChainInfo(const FieldChainInfo &Other,
                               const FieldRegion *FR, const bool IsDereferenced)
    : FieldChainInfo(Other, IsDereferenced) {
  assert(!contains(FR) && "Can't add a field that is already a part of the "
                          "fieldchain! Is this a cyclic reference?");
  Chain = Factory.add(FR, Other.Chain);
}

bool FieldChainInfo::isPointer() const {
  assert(!Chain.isEmpty() && "Empty fieldchain!");
  return (*Chain.begin())->getDecl()->getType()->isPointerType();
}

bool FieldChainInfo::isDereferenced() const {
  assert(isPointer() && "Only pointers may or may not be dereferenced!");
  return IsDereferenced;
}

const FieldDecl *FieldChainInfo::getEndOfChain() const {
  assert(!Chain.isEmpty() && "Empty fieldchain!");
  return (*Chain.begin())->getDecl();
}

// TODO: This function constructs an incorrect string if a void pointer is a
// part of the chain:
//
//   struct B { int x; }
//
//   struct A {
//     void *vptr;
//     A(void* vptr) : vptr(vptr) {}
//   };
//
//   void f() {
//     B b;
//     A a(&b);
//   }
//
// The note message will be "uninitialized field 'this->vptr->x'", even though
// void pointers can't be dereferenced. This should be changed to "uninitialized
// field 'static_cast<B*>(this->vptr)->x'".
//
// TODO: This function constructs an incorrect fieldchain string in the
// following case:
//
//   struct Base { int x; };
//   struct D1 : Base {}; struct D2 : Base {};
//
//   struct MostDerived : D1, D2 {
//     MostDerived() {}
//   }
//
// A call to MostDerived::MostDerived() will cause two notes that say
// "uninitialized field 'this->x'", but we can't refer to 'x' directly,
// we need an explicit namespace resolution whether the uninit field was
// 'D1::x' or 'D2::x'.
void FieldChainInfo::print(llvm::raw_ostream &Out) const {
  if (Chain.isEmpty())
    return;

  const llvm::ImmutableListImpl<const FieldRegion *> *L =
      Chain.getInternalPointer();
  printTail(Out, L->getTail());
  Out << getVariableName(L->getHead()->getDecl());
}

void FieldChainInfo::printTail(
    llvm::raw_ostream &Out,
    const llvm::ImmutableListImpl<const FieldRegion *> *L) {
  if (!L)
    return;

  printTail(Out, L->getTail());
  const FieldDecl *Field = L->getHead()->getDecl();
  Out << getVariableName(Field);
  Out << (Field->getType()->isPointerType() ? "->" : ".");
}

//===----------------------------------------------------------------------===//
//                           Utility functions.
//===----------------------------------------------------------------------===//

static Optional<nonloc::LazyCompoundVal>
getObjectVal(const CXXConstructorDecl *CtorDecl, CheckerContext &Context) {

  Loc ThisLoc = Context.getSValBuilder().getCXXThis(CtorDecl->getParent(),
                                                    Context.getStackFrame());
  // Getting the value for 'this'.
  SVal This = Context.getState()->getSVal(ThisLoc);

  // Getting the value for '*this'.
  SVal Object = Context.getState()->getSVal(This.castAs<Loc>());

  return Object.getAs<nonloc::LazyCompoundVal>();
}

static bool willObjectBeAnalyzedLater(const CXXConstructorDecl *Ctor,
                               CheckerContext &Context) {

  Optional<nonloc::LazyCompoundVal> CurrentObject = getObjectVal(Ctor, Context);
  if (!CurrentObject)
    return false;

  const LocationContext *LC = Context.getLocationContext();
  while ((LC = LC->getParent())) {

    // If \p Ctor was called by another constructor.
    const auto *OtherCtor = dyn_cast<CXXConstructorDecl>(LC->getDecl());
    if (!OtherCtor)
      continue;

    Optional<nonloc::LazyCompoundVal> OtherObject =
        getObjectVal(OtherCtor, Context);
    if (!OtherObject)
      continue;

    // If the CurrentObject is a subregion of OtherObject, it will be analyzed
    // during the analysis of OtherObject.
    if (CurrentObject->getRegion()->isSubRegionOf(OtherObject->getRegion()))
      return true;
  }

  return false;
}

static void printNoteMessage(llvm::raw_ostream &Out,
                             const FieldChainInfo &Chain) {
  if (Chain.isPointer()) {
    if (Chain.isDereferenced())
      Out << "uninitialized pointee 'this->";
    else
      Out << "uninitialized pointer 'this->";
  } else
    Out << "uninitialized field 'this->";
  Chain.print(Out);
  Out << "'";
}

static StringRef getVariableName(const FieldDecl *Field) {
  // If Field is a captured lambda variable, Field->getName() will return with
  // an empty string. We can however acquire it's name from the lambda's
  // captures.
  const auto *CXXParent = dyn_cast<CXXRecordDecl>(Field->getParent());

  if (CXXParent && CXXParent->isLambda()) {
    assert(CXXParent->captures_begin());
    auto It = CXXParent->captures_begin() + Field->getFieldIndex();
    return It->getCapturedVar()->getName();
  }

  return Field->getName();
}

void ento::registerUninitializedObjectChecker(CheckerManager &Mgr) {
  auto Chk = Mgr.registerChecker<UninitializedObjectChecker>();
  Chk->IsPedantic = Mgr.getAnalyzerOptions().getBooleanOption(
      "Pedantic", /*DefaultVal*/ false, Chk);
  Chk->ShouldConvertNotesToWarnings = Mgr.getAnalyzerOptions().getBooleanOption(
      "NotesAsWarnings", /*DefaultVal*/ false, Chk);
  Chk->CheckPointeeInitialization = Mgr.getAnalyzerOptions().getBooleanOption(
      "CheckPointeeInitialization", /*DefaultVal*/ false, Chk);
}
