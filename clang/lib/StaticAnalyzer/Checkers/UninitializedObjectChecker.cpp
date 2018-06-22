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
// This checker has an option "Pedantic" (boolean). If its not set or is set to
// false, the checker won't emit warnings for objects that don't have at least
// one initialized field. This may be set with
// `-analyzer-config alpha.cplusplus.UninitializedObject:Pedantic=true`.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include <algorithm>

using namespace clang;
using namespace clang::ento;

namespace {

class UninitializedObjectChecker : public Checker<check::EndFunction> {
  std::unique_ptr<BuiltinBug> BT_uninitField;

public:
  bool IsPedantic; // Will be initialized when registering the checker.

  UninitializedObjectChecker()
      : BT_uninitField(new BuiltinBug(this, "Uninitialized fields")) {}
  void checkEndFunction(CheckerContext &C) const;
};

llvm::ImmutableListFactory<const FieldRegion *> Factory;

/// Represents a field chain. A field chain is a vector of fields where the
/// first element of the chain is the object under checking (not stored), and
/// every other element is a field, and the element that precedes it is the
/// object that contains it.
///
/// Note that this class is immutable, and new fields may only be added through
/// constructor calls.
class FieldChainInfo {
  using FieldChain = llvm::ImmutableList<const FieldRegion *>;

  FieldChain Chain;

  const bool IsDereferenced = false;

public:
  FieldChainInfo() = default;

  FieldChainInfo(const FieldChainInfo &Other, const bool IsDereferenced)
      : Chain(Other.Chain), IsDereferenced(IsDereferenced) {}

  FieldChainInfo(const FieldChainInfo &Other, const FieldRegion *FR,
                 const bool IsDereferenced = false);

  bool contains(const FieldRegion *FR) const { return Chain.contains(FR); }
  bool isPointer() const;

  /// If this is a fieldchain whose last element is an uninitialized region of a
  /// pointer type, `IsDereferenced` will store whether the pointer itself or
  /// the pointee is uninitialized.
  bool isDereferenced() const;
  const FieldDecl *getEndOfChain() const;
  void print(llvm::raw_ostream &Out) const;

private:
  /// Prints every element except the last to `Out`. Since ImmutableLists store
  /// elements in reverse order, and have no reverse iterators, we use a
  /// recursive function to print the fieldchain correctly. The last element in
  /// the chain is to be printed by `print`.
  static void printTail(llvm::raw_ostream &Out,
                        const llvm::ImmutableListImpl<const FieldRegion *> *L);
  friend struct FieldChainInfoComparator;
};

struct FieldChainInfoComparator {
  bool operator()(const FieldChainInfo &lhs, const FieldChainInfo &rhs) const {
    assert(!lhs.Chain.isEmpty() && !rhs.Chain.isEmpty() &&
           "Attempted to store an empty fieldchain!");
    return *lhs.Chain.begin() < *rhs.Chain.begin();
  }
};

using UninitFieldSet = std::set<FieldChainInfo, FieldChainInfoComparator>;

/// Searches for and stores uninitialized fields in a non-union object.
class FindUninitializedFields {
  ProgramStateRef State;
  const TypedValueRegion *const ObjectR;

  const bool IsPedantic;
  bool IsAnyFieldInitialized = false;

  UninitFieldSet UninitFields;

public:
  FindUninitializedFields(ProgramStateRef State,
                          const TypedValueRegion *const R, bool IsPedantic);
  const UninitFieldSet &getUninitFields();

private:
  /// Adds a FieldChainInfo object to UninitFields. Return true if an insertion
  /// took place.
  bool addFieldToUninits(FieldChainInfo LocalChain);

  // For the purposes of this checker, we'll regard the object under checking as
  // a directed tree, where
  //   * the root is the object under checking
  //   * every node is an object that is
  //     - a union
  //     - a non-union record
  //     - a pointer/reference
  //     - an array
  //     - of a member pointer type
  //     - of a primitive type, which we'll define as either a BuiltinType or
  //       EnumeralType.
  //   * the parent of each node is the object that contains it
  //   * every leaf is an array, a primitive object, a member pointer, a nullptr
  //     or an undefined pointer.
  //
  // Example:
  //
  //   struct A {
  //      struct B {
  //        int x, y = 0;
  //      };
  //      B b;
  //      int *iptr = new int;
  //      B* bptr;
  //
  //      A() {}
  //   };
  //
  // The directed tree:
  //
  //           ->x
  //          /
  //      ->b--->y
  //     /
  //    A-->iptr->(int value)
  //     \
  //      ->bptr
  //
  // From this we'll construct a vector of fieldchains, where each fieldchain
  // represents an uninitialized field. An uninitialized field may be a
  // primitive object, a member pointer, a pointer, a pointee or a union without
  // a single initialized field.
  // In the above example, for the default constructor call we'll end up with
  // these fieldchains:
  //
  //   this->b.x
  //   this->iptr (pointee uninit)
  //   this->bptr (pointer uninit)
  //
  // We'll traverse each node of the above graph with the appropiate one of
  // these methods:

  /// This method checks a region of a union object, and returns true if no
  /// field is initialized within the region.
  bool isUnionUninit(const TypedValueRegion *R);

  /// This method checks a region of a non-union object, and returns true if
  /// an uninitialized field is found within the region.
  bool isNonUnionUninit(const TypedValueRegion *R, FieldChainInfo LocalChain);

  /// This method checks a region of a pointer or reference object, and returns
  /// true if the ptr/ref object itself or any field within the pointee's region
  /// is uninitialized.
  bool isPointerOrReferenceUninit(const FieldRegion *FR,
                                  FieldChainInfo LocalChain);

  /// This method checks a region of MemberPointerType, and returns true if the
  /// the pointer is uninitialized.
  bool isMemberPointerUninit(const FieldRegion *FR, FieldChainInfo LocalChain);

  /// This method returns true if the value of a primitive object is
  /// uninitialized.
  bool isPrimitiveUninit(const SVal &V);

  // Note that we don't have a method for arrays -- the elements of an array are
  // often left uninitialized intentionally even when it is of a C++ record
  // type, so we'll assume that an array is always initialized.
  // TODO: Add a support for nonloc::LocAsInteger.
};

// Utility function declarations.

/// Returns the object that was constructed by CtorDecl, or None if that isn't
/// possible.
Optional<nonloc::LazyCompoundVal>
getObjectVal(const CXXConstructorDecl *CtorDecl, CheckerContext &Context);

/// Checks whether the constructor under checking is called by another
/// constructor.
bool isCalledByConstructor(const CheckerContext &Context);

/// Returns whether FD can be (transitively) dereferenced to a void pointer type
/// (void*, void**, ...). The type of the region behind a void pointer isn't
/// known, and thus FD can not be analyzed.
bool isVoidPointer(const FieldDecl *FD);

/// Returns true if T is a primitive type. We'll call a type primitive if it's
/// either a BuiltinType or an EnumeralType.
bool isPrimitiveType(const QualType &T) {
  return T->isBuiltinType() || T->isEnumeralType();
}

} // end of anonymous namespace

//===----------------------------------------------------------------------===//
//                  Methods for UninitializedObjectChecker.
//===----------------------------------------------------------------------===//

void UninitializedObjectChecker::checkEndFunction(
    CheckerContext &Context) const {

  const auto *CtorDecl = dyn_cast_or_null<CXXConstructorDecl>(
      Context.getLocationContext()->getDecl());
  if (!CtorDecl)
    return;

  if (!CtorDecl->isUserProvided())
    return;

  if (CtorDecl->getParent()->isUnion())
    return;

  // This avoids essentially the same error being reported multiple times.
  if (isCalledByConstructor(Context))
    return;

  Optional<nonloc::LazyCompoundVal> Object = getObjectVal(CtorDecl, Context);
  if (!Object)
    return;

  FindUninitializedFields F(Context.getState(), Object->getRegion(),
                            IsPedantic);

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

  SmallString<100> WarningBuf;
  llvm::raw_svector_ostream WarningOS(WarningBuf);
  WarningOS << UninitFields.size() << " uninitialized field"
            << (UninitFields.size() == 1 ? "" : "s")
            << " at the end of the constructor call";

  auto Report = llvm::make_unique<BugReport>(
      *BT_uninitField, WarningOS.str(), Node, LocUsedForUniqueing,
      Node->getLocationContext()->getDecl());

  // TODO: As of now, one warning is emitted per constructor call, and the
  // uninitialized fields are listed in notes. Until there's a better support
  // for notes avaible, a note-less version of this checker should be
  // implemented.
  for (const auto &FieldChain : UninitFields) {
    SmallString<200> NoteBuf;
    llvm::raw_svector_ostream NoteOS(NoteBuf);

    if (FieldChain.isPointer()) {
      if (FieldChain.isDereferenced())
        NoteOS << "uninitialized pointee 'this->";
      else
        NoteOS << "uninitialized pointer 'this->";
    } else
      NoteOS << "uninitialized field 'this->";
    FieldChain.print(NoteOS);
    NoteOS << "'";

    Report->addNote(NoteOS.str(),
                    PathDiagnosticLocation::create(FieldChain.getEndOfChain(),
                                                   Context.getSourceManager()));
  }

  Context.emitReport(std::move(Report));
}

//===----------------------------------------------------------------------===//
//                   Methods for FindUninitializedFields.
//===----------------------------------------------------------------------===//

FindUninitializedFields::FindUninitializedFields(
    ProgramStateRef State, const TypedValueRegion *const R, bool IsPedantic)
    : State(State), ObjectR(R), IsPedantic(IsPedantic) {}

const UninitFieldSet &FindUninitializedFields::getUninitFields() {
  isNonUnionUninit(ObjectR, FieldChainInfo());

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

    if (T->isMemberPointerType()) {
      if (isMemberPointerUninit(FR, LocalChain))
        ContainsUninitField = true;
      continue;
    }

    // If this is a pointer or reference type.
    if (T->isPointerType() || T->isReferenceType()) {
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
  // FIXME: As of now, because of `isCalledByConstructor`, objects whose type
  // is a descendant of another type will emit warnings for uninitalized
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

// Note that pointers/references don't contain fields themselves, so in this
// function we won't add anything to LocalChain.
bool FindUninitializedFields::isPointerOrReferenceUninit(
    const FieldRegion *FR, FieldChainInfo LocalChain) {

  assert((FR->getDecl()->getType()->isPointerType() ||
          FR->getDecl()->getType()->isReferenceType()) &&
         "This method only checks pointer/reference objects!");

  SVal V = State->getSVal(FR);

  if (V.isUnknown() || V.isZeroConstant()) {
    IsAnyFieldInitialized = true;
    return false;
  }

  if (V.isUndef()) {
    return addFieldToUninits({LocalChain, FR});
  }

  const FieldDecl *FD = FR->getDecl();

  // TODO: The dynamic type of a void pointer may be retrieved with
  // `getDynamicTypeInfo`.
  if (isVoidPointer(FD)) {
    IsAnyFieldInitialized = true;
    return false;
  }

  assert(V.getAs<Loc>() && "V should be Loc at this point!");

  // At this point the pointer itself is initialized and points to a valid
  // location, we'll now check the pointee.
  SVal DerefdV = State->getSVal(V.castAs<Loc>());

  // TODO: Dereferencing should be done according to the dynamic type.
  while (Optional<Loc> L = DerefdV.getAs<Loc>()) {
    DerefdV = State->getSVal(*L);
  }

  // If V is a pointer pointing to a record type.
  if (Optional<nonloc::LazyCompoundVal> RecordV =
          DerefdV.getAs<nonloc::LazyCompoundVal>()) {

    const TypedValueRegion *R = RecordV->getRegion();

    // We can't reason about symbolic regions, assume its initialized.
    // Note that this also avoids a potential infinite recursion, because
    // constructors for list-like classes are checked without being called, and
    // the Static Analyzer will construct a symbolic region for Node *next; or
    // similar code snippets.
    if (R->getSymbolicBase()) {
      IsAnyFieldInitialized = true;
      return false;
    }

    const QualType T = R->getValueType();

    if (T->isStructureOrClassType())
      return isNonUnionUninit(R, {LocalChain, FR});

    if (T->isUnionType()) {
      if (isUnionUninit(R)) {
        return addFieldToUninits({LocalChain, FR, /*IsDereferenced*/ true});
      } else {
        IsAnyFieldInitialized = true;
        return false;
      }
    }

    if (T->isArrayType()) {
      IsAnyFieldInitialized = true;
      return false;
    }

    llvm_unreachable("All cases are handled!");
  }

  // TODO: If possible, it should be asserted that the DerefdV at this point is
  // primitive.

  if (isPrimitiveUninit(DerefdV))
    return addFieldToUninits({LocalChain, FR, /*IsDereferenced*/ true});

  IsAnyFieldInitialized = true;
  return false;
}

bool FindUninitializedFields::isMemberPointerUninit(const FieldRegion *FR,
                                                    FieldChainInfo LocalChain) {
  assert(FR->getDecl()->getType()->isMemberPointerType() &&
         "This function only checks regions that hold MemberPointerTypes!");
  // TODO: Implement support for MemberPointerTypes.
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
//
// TODO: If a field in the fieldchain is a captured lambda parameter, this
// function constructs an empty string for it:
//
//   template <class Callable> struct A {
//     Callable c;
//     A(const Callable &c, int) : c(c) {}
//   };
//
//   int b; // say that this isn't zero initialized
//   auto alwaysTrue = [&b](int a) { return true; };
//
// A call with these parameters: A<decltype(alwaysTrue)>::A(alwaysTrue, int())
// will emit a note with the message "uninitialized field: 'this->c.'". If
// possible, the lambda parameter name should be retrieved or be replaced with a
// "<lambda parameter>" or something similar.
void FieldChainInfo::print(llvm::raw_ostream &Out) const {
  if (Chain.isEmpty())
    return;

  const llvm::ImmutableListImpl<const FieldRegion *> *L =
      Chain.getInternalPointer();
  printTail(Out, L->getTail());
  Out << L->getHead()->getDecl()->getNameAsString();
}

void FieldChainInfo::printTail(
    llvm::raw_ostream &Out,
    const llvm::ImmutableListImpl<const FieldRegion *> *L) {
  if (!L)
    return;

  printTail(Out, L->getTail());
  const FieldDecl *Field = L->getHead()->getDecl();
  Out << Field->getNameAsString();
  Out << (Field->getType()->isPointerType() ? "->" : ".");
}

//===----------------------------------------------------------------------===//
//                           Utility functions.
//===----------------------------------------------------------------------===//

namespace {

bool isVoidPointer(const FieldDecl *FD) {
  QualType T = FD->getType();

  while (!T.isNull()) {
    if (T->isVoidPointerType())
      return true;
    T = T->getPointeeType();
  }
  return false;
}

Optional<nonloc::LazyCompoundVal>
getObjectVal(const CXXConstructorDecl *CtorDecl, CheckerContext &Context) {

  Loc ThisLoc = Context.getSValBuilder().getCXXThis(CtorDecl->getParent(),
                                                    Context.getStackFrame());
  // Getting the value for 'this'.
  SVal This = Context.getState()->getSVal(ThisLoc);

  // Getting the value for '*this'.
  SVal Object = Context.getState()->getSVal(This.castAs<Loc>());

  return Object.getAs<nonloc::LazyCompoundVal>();
}

// TODO: We should also check that if the constructor was called by another
// constructor, whether those two are in any relation to one another. In it's
// current state, this introduces some false negatives.
bool isCalledByConstructor(const CheckerContext &Context) {
  const LocationContext *LC = Context.getLocationContext()->getParent();

  while (LC) {
    if (isa<CXXConstructorDecl>(LC->getDecl()))
      return true;

    LC = LC->getParent();
  }
  return false;
}

} // end of anonymous namespace

void ento::registerUninitializedObjectChecker(CheckerManager &Mgr) {
  auto Chk = Mgr.registerChecker<UninitializedObjectChecker>();
  Chk->IsPedantic = Mgr.getAnalyzerOptions().getBooleanOption(
      "Pedantic", /*DefaultVal*/ false, Chk);
}
