//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnsequencedGlobalAccessesCheck.h"

#include "../utils/ExecutionVisitor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
// An AccesKind represents one access to a global variable.
//
// The unchecked versions represent reads/writes that are not handled by
// -Wunsequenced. (e.g. accesses inside functions).
enum AccessKind : uint8_t { AkRead = 0, AkWrite, AkLast };

static constexpr uint8_t AkCount = AkLast;

// The TraversalResultKind represents a set of accesses.
// Bits are corresponding to the AccessKind enum values.
using TraversalResultKind = uint8_t;
static constexpr TraversalResultKind TrInvalid = 0;
static constexpr TraversalResultKind TrRead = 1 << AkRead;
static constexpr TraversalResultKind TrWrite = 1 << AkWrite;

// To represent fields in structs or unions we use numbered FieldIndices. The
// FieldIndexArray represents one field inside a global struct/union system.
// The FieldIndexArray can be thought of as a path inside a tree.
using FieldIndex = uint16_t;
static constexpr FieldIndex FiUnion = 0x8000;

// Note: This bit signals whether the field is a *field of* a struct or a
// union, not whether the type of the field itself is a struct or a union.
using FieldIndexArray = SmallVector<FieldIndex>;

/// One traversal recurses into one side of a binary expression or one
/// parameter of a function call. At least two of these traversals are used to
/// find conflicting accesses.
///
/// A TraversalResult represents one traversal.
struct TraversalResult {
  int IndexCreated; // We use indices to keep track of which
                    // traversal we are in currently. The current
                    // index is stored in GlobalRWVisitor with the
                    // name TraversalIndex.
  SourceLocation Loc[AkCount];
  TraversalResultKind Kind;

  TraversalResult();
  TraversalResult(int Index, SourceLocation Loc, AccessKind Access);
  void addNewAccess(SourceLocation Loc, AccessKind Access);
};

/// The result of a number of traversals.
class TraversalAggregation {
  DeclarationName DeclName; // The name of the global variable being checked.

  // We only store the result of two traversals as two conflicting accesses
  // are enough to detect undefined behavior. The two stored TraversalResults
  // have different traversal indices.
  //
  // Note: Sometimes multiple traversals are merged into one
  // TraversalResult.
  TraversalResult MainPart, OtherPart;
  // Double reads are not reportable.

public:
  TraversalAggregation();
  TraversalAggregation(DeclarationName Name, SourceLocation Loc,
                       AccessKind Access, int Index);
  void addGlobalRW(SourceLocation Loc, AccessKind Access, int Index);
  DeclarationName getDeclName() const;

  bool isValid() const;

  // If there is a conflict and that conflict isn't reported by -Wunsequenced
  // then we report the conflict.
  bool shouldBeReported() const;
  bool hasConflictingOperations() const;

private:
  bool hasTwoAccesses() const;
};

/// The ObjectAccessTree stores the TraversalAggregations of one global
/// struct/union. Because each field can be handled as a single variable, the
/// tree stores one TraversalAggregation for every field.
///
/// Note: structs, classes, and unions are called objects in the code.
struct ObjectAccessTree {
  using FieldMap = llvm::DenseMap<int, std::unique_ptr<ObjectAccessTree>>;
  TraversalAggregation OwnAccesses;

  // In a union, new fields should inherit from UnionTemporalAccesses
  // instead of OwnAccesses. That's because an access to a field of a union is
  // also an access to every other field of the same union.
  TraversalAggregation UnionTemporalAccesses;

  // We try to be lazy and only store fields that are actually accessed.
  FieldMap Fields;
  bool IsUnion;

  ObjectAccessTree(TraversalAggregation Own);

  void addFieldToAll(SourceLocation Loc, AccessKind Access, int Index);
  void addFieldToAllExcept(uint16_t ExceptIndex, SourceLocation Loc,
                           AccessKind Access, int Index);
};

/// This object is the root of all ObjectAccessTrees.
class ObjectTraversalAggregation {
  DeclarationName DeclName; // The name of the global struct/union.
  ObjectAccessTree AccessTree;

public:
  ObjectTraversalAggregation(DeclarationName Name, SourceLocation Loc,
                             FieldIndexArray FieldIndices, AccessKind Access,
                             int Index);
  void addFieldRW(SourceLocation Loc, FieldIndexArray FieldIndices,
                  AccessKind Access, int Index);
  DeclarationName getDeclName() const;
  bool shouldBeReported() const;

private:
  bool shouldBeReportedRec(const ObjectAccessTree *Node) const;
};

using utils::ExecutionVisitor;

/// GlobalRWVisitor (global read write visitor) does all the traversals.
class GlobalRWVisitor : public ExecutionVisitor<GlobalRWVisitor> {
public:
  GlobalRWVisitor(bool IsWritePossibleThroughFunctionParam);

  // startTraversal is called to start a new traversal. It increments the
  // TraversalIndex, which in turn will generate new TraversalResults.
  void startTraversal(const Expr *E);

  const llvm::SmallVector<TraversalAggregation> &getGlobalsFound() const;

  const llvm::SmallVector<ObjectTraversalAggregation> &
  getObjectGlobalsFound() const;

  // RecursiveASTVisitor overrides
  bool VisitDeclRefExpr(DeclRefExpr *S);
  bool VisitUnaryOperator(UnaryOperator *Op);
  bool VisitBinaryOperator(BinaryOperator *Op);
  bool VisitCallExpr(CallExpr *CE);
  bool VisitCXXConstructExpr(CXXConstructExpr *CE);
  bool VisitMemberExpr(MemberExpr *ME);

private:
  void visitFunctionLikeExprArgs(const FunctionProtoType *FT,
                                 CallExpr::const_arg_range Arguments);
  void visitCallExprArgs(const CallExpr *CE);
  void visitConstructExprArgs(const CXXConstructExpr *CE);

  llvm::SmallVector<TraversalAggregation> GlobalsFound;
  llvm::SmallVector<ObjectTraversalAggregation> ObjectGlobalsFound;

  // The TraversalIndex is used to differentiate between two sides of a binary
  // expression or the parameters of a function. Every traversal represents
  // one such expression and the TraversalIndex is incremented between them.
  int TraversalIndex;

  // Same as the HandleMutableFunctionParametersAsWrites option.
  bool IsWritePossibleThroughFunctionParam;

  void addGlobal(DeclarationName Name, SourceLocation Loc, bool IsWrite);
  void addGlobal(const DeclRefExpr *DR, bool IsWrite);
  void addField(DeclarationName Name, FieldIndexArray FieldIndices,
                SourceLocation Loc, bool IsWrite);
  bool handleModified(const Expr *Modified);
  bool handleModifiedVariable(const DeclRefExpr *DE);
  bool handleAccessedObject(const Expr *E, bool IsWrite);
  bool isVariable(const Expr *E);
};

AST_MATCHER_P(BinaryOperator, unsequencedBinaryOperator, const LangStandard *,
              LangStd) {
  assert(LangStd);

  const BinaryOperator *Op = &Node;

  const BinaryOperator::Opcode Code = Op->getOpcode();
  if (Code == BO_LAnd || Code == BO_LOr || Code == BO_Comma)
    return false;

  if (Op->isAssignmentOp() && isa<DeclRefExpr>(Op->getLHS()))
    return false;

  if (LangStd->isCPlusPlus17() &&
      (Code == BO_Shl || Code == BO_Shr || Code == BO_PtrMemD ||
       Code == BO_PtrMemI || Op->isAssignmentOp()))
    return false;

  return true;
}
} // namespace

static bool isGlobalDecl(const VarDecl *VD) {
  return VD && VD->hasGlobalStorage() && VD->getLocation().isValid() &&
         !VD->getType().isConstQualified();
}

UnsequencedGlobalAccessesCheck::UnsequencedGlobalAccessesCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      HandleMutableFunctionParametersAsWrites(
          Options.get("HandleMutableFunctionParametersAsWrites", false)) {}

void UnsequencedGlobalAccessesCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HandleMutableFunctionParametersAsWrites",
                HandleMutableFunctionParametersAsWrites);
}

void UnsequencedGlobalAccessesCheck::registerMatchers(MatchFinder *Finder) {

  const LangStandard *LangStd =
      &LangStandard::getLangStandardForKind(getLangOpts().LangStd);

  auto BinaryOperatorMatcher = unsequencedBinaryOperator(LangStd);

  Finder->addMatcher(
      stmt(traverse(TK_AsIs, binaryOperator(BinaryOperatorMatcher).bind("gw"))),
      this);

  // Array subscript expressions became sequenced in C++17
  if (!LangStd->isCPlusPlus17())
    Finder->addMatcher(stmt(traverse(TK_AsIs, arraySubscriptExpr().bind("gw"))),
                       this);

  Finder->addMatcher(stmt(traverse(TK_AsIs, callExpr().bind("gw"))), this);

  if (!LangStd->isCPlusPlus11())
    Finder->addMatcher(stmt(traverse(TK_AsIs, initListExpr().bind("gw"))),
                       this);

  Finder->addMatcher(stmt(traverse(TK_AsIs, cxxConstructExpr().bind("gw"))),
                     this);
}

void UnsequencedGlobalAccessesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const Expr *E = Result.Nodes.getNodeAs<Expr>("gw");

  GlobalRWVisitor Visitor(HandleMutableFunctionParametersAsWrites);
  if (const auto *Op = dyn_cast<BinaryOperator>(E)) {
    Visitor.startTraversal(Op->getLHS());
    Visitor.startTraversal(Op->getRHS());

  } else if (const auto *CE = dyn_cast<CallExpr>(E)) {
    // A CallExpr will include its defaulted arguments as well.
    for (const Expr *Arg : CE->arguments()) {
      // For some reason, calling TraverseStmt on Arg directly
      // doesn't recurse when Arg is a default argument.
      if (const auto *DA = dyn_cast<CXXDefaultArgExpr>(Arg)) {
        Visitor.startTraversal(DA->getExpr());
        continue;
      }
      Visitor.startTraversal(Arg);
    }
  } else if (const auto *AS = dyn_cast<ArraySubscriptExpr>(E)) {
    Visitor.startTraversal(AS->getLHS());
    Visitor.startTraversal(AS->getRHS());
  } else if (const auto *IL = dyn_cast<InitListExpr>(E)) {
    llvm::SmallVector<const InitListExpr *> NestedInitializers;
    NestedInitializers.push_back(IL);
    while (!NestedInitializers.empty()) {
      const InitListExpr *CurrentIL =
          NestedInitializers[NestedInitializers.size() - 1];
      NestedInitializers.pop_back();

      for (const auto *I : CurrentIL->inits()) {
        if (const InitListExpr *Nested = dyn_cast<InitListExpr>(I)) {
          NestedInitializers.push_back(Nested);
          continue;
        }

        Visitor.startTraversal(I);
      }
    }
  } else if (const auto *CE = dyn_cast<CXXConstructExpr>(E)) {
    for (const Expr *Arg : CE->arguments()) {
      // For some reason, calling TraverseStmt on Arg directly
      // doesn't recurse when Arg is a default argument.
      if (const auto *DA = dyn_cast<CXXDefaultArgExpr>(Arg)) {
        Visitor.startTraversal(DA->getExpr());
        continue;
      }
      Visitor.startTraversal(Arg);
    }
  }

  const llvm::SmallVector<TraversalAggregation> &Globals =
      Visitor.getGlobalsFound();

  for (const TraversalAggregation &Global : Globals)
    if (Global.shouldBeReported())
      diag(E->getBeginLoc(), "read/write conflict on global variable " +
                                 Global.getDeclName().getAsString());

  const llvm::SmallVector<ObjectTraversalAggregation> &ObjectGlobals =
      Visitor.getObjectGlobalsFound();

  for (const ObjectTraversalAggregation &ObjectGlobal : ObjectGlobals)
    if (ObjectGlobal.shouldBeReported())
      diag(E->getBeginLoc(), "read/write conflict on the field of the global "
                             "object " +
                                 ObjectGlobal.getDeclName().getAsString());
}

GlobalRWVisitor::GlobalRWVisitor(bool IsWritePossibleThroughFunctionParam)
    : TraversalIndex(0),
      IsWritePossibleThroughFunctionParam(IsWritePossibleThroughFunctionParam) {
}

void GlobalRWVisitor::startTraversal(const Expr *E) {
  TraversalIndex++;
  traverseExecution(const_cast<Expr *>(E));
}

bool GlobalRWVisitor::isVariable(const Expr *E) {
  const Type *T = E->getType().getTypePtrOrNull();
  if (!T)
    return false;

  return isa<DeclRefExpr>(E) && (!T->isRecordType() || T->isUnionType());
}

bool GlobalRWVisitor::VisitDeclRefExpr(DeclRefExpr *DR) {
  const auto *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return true;

  if (!isVariable(DR))
    return handleAccessedObject(DR, /*IsWrite*/ false);

  if (isGlobalDecl(VD)) {
    addGlobal(VD->getDeclName(), VD->getBeginLoc(), /*IsWrite*/ false);
    return true;
  }
  return true;
}

bool GlobalRWVisitor::VisitMemberExpr(MemberExpr *ME) {
  return handleAccessedObject(ME, /*IsWrite*/ false);
}

bool GlobalRWVisitor::handleModifiedVariable(const DeclRefExpr *DR) {
  const auto *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return true;

  if (isGlobalDecl(VD)) {
    addGlobal(VD->getDeclName(), VD->getBeginLoc(), /*IsWrite*/ true);
    return false;
  }

  return true;
}

bool GlobalRWVisitor::handleAccessedObject(const Expr *E, bool IsWrite) {
  const Expr *CurrentNode = E;
  int NodeCount = 0;
  while (isa<MemberExpr>(CurrentNode)) {
    const MemberExpr *CurrentField = dyn_cast<MemberExpr>(CurrentNode);

    if (CurrentField->isArrow())
      return true;

    const ValueDecl *Decl = CurrentField->getMemberDecl();
    if (!isa<FieldDecl>(Decl))
      return true;

    CurrentNode = CurrentField->getBase();
    NodeCount++;
  }

  const DeclRefExpr *Base = dyn_cast<DeclRefExpr>(CurrentNode);
  if (!Base)
    return true;

  const VarDecl *BaseDecl = dyn_cast<VarDecl>(Base->getDecl());
  if (!BaseDecl)
    return true;

  if (!isGlobalDecl(BaseDecl))
    return true;

  FieldIndexArray FieldIndices(NodeCount);
  CurrentNode = E;
  while (isa<MemberExpr>(CurrentNode)) {
    const MemberExpr *CurrentField = dyn_cast<MemberExpr>(CurrentNode);
    const FieldDecl *Decl = dyn_cast<FieldDecl>(CurrentField->getMemberDecl());
    assert(Decl);

    FieldIndices[NodeCount - 1] = Decl->getFieldIndex();
    const RecordDecl *Record = Decl->getParent();
    assert(Record);

    if (Record->isUnion())
      FieldIndices[NodeCount - 1] |= FiUnion;

    CurrentNode = CurrentField->getBase();
    NodeCount--;
  }

  addField(BaseDecl->getDeclName(), FieldIndices, Base->getBeginLoc(), IsWrite);
  return false;
}

bool GlobalRWVisitor::handleModified(const Expr *Modified) {
  assert(Modified);

  if (isVariable(Modified))
    return handleModifiedVariable(dyn_cast<DeclRefExpr>(Modified));

  return handleAccessedObject(Modified, /*IsWrite*/ true);
}

bool GlobalRWVisitor::VisitUnaryOperator(UnaryOperator *Op) {
  UnaryOperator::Opcode Code = Op->getOpcode();
  if (Code == UO_PostInc || Code == UO_PostDec || Code == UO_PreInc ||
      Code == UO_PreDec)
    return handleModified(Op->getSubExpr());

  // Ignore the AddressOf operator as it doesn't read the variable.
  if (Code == UO_AddrOf && isa<DeclRefExpr>(Op->getSubExpr()))
    return false;

  return true;
}

bool GlobalRWVisitor::VisitBinaryOperator(BinaryOperator *Op) {
  if (Op->isAssignmentOp())
    return handleModified(Op->getLHS());

  return true;
}

void GlobalRWVisitor::visitFunctionLikeExprArgs(
    const FunctionProtoType *FT, CallExpr::const_arg_range Arguments) {

  uint32_t I = 0;
  auto ArgumentsEnd = Arguments.end();
  for (auto It = Arguments.begin(); It != ArgumentsEnd; It++, I++) {
    const Expr *Arg = *It;

    if (I >= FT->getNumParams())
      continue;

    if (const auto *Op = dyn_cast<UnaryOperator>(Arg)) {
      if (Op->getOpcode() != UO_AddrOf)
        continue;

      if (const auto *PtrType = dyn_cast_if_present<PointerType>(
              FT->getParamType(I).getTypePtrOrNull())) {
        if (PtrType->getPointeeType().isConstQualified())
          continue;

        if (handleModified(Op->getSubExpr()))
          continue;
      }
    }

    if (const auto *RefType = dyn_cast_if_present<ReferenceType>(
            FT->getParamType(I).getTypePtrOrNull())) {
      if (RefType->getPointeeType().isConstQualified())
        continue;

      if (handleModified(Arg))
        continue;
    }
  }
}

void GlobalRWVisitor::visitCallExprArgs(const CallExpr *CE) {
  const Type *CT = CE->getCallee()->getType().getTypePtrOrNull();
  if (const auto *PT = dyn_cast_if_present<PointerType>(CT))
    CT = PT->getPointeeType().getTypePtrOrNull();

  const auto *ProtoType = dyn_cast_if_present<FunctionProtoType>(CT);
  if (!ProtoType)
    return;

  visitFunctionLikeExprArgs(ProtoType, CE->arguments());
}

void GlobalRWVisitor::visitConstructExprArgs(const CXXConstructExpr *E) {
  const FunctionDecl *Decl = E->getConstructor();
  const Type *T = Decl->getType().getTypePtrOrNull();
  if (!T)
    return;

  const auto *FT = dyn_cast<FunctionProtoType>(T);
  if (!FT)
    return;

  visitFunctionLikeExprArgs(FT, E->arguments());
}

bool GlobalRWVisitor::VisitCallExpr(CallExpr *CE) {

  if (IsWritePossibleThroughFunctionParam || isa<CXXOperatorCallExpr>(CE))
    visitCallExprArgs(CE);

  return ExecutionVisitor::VisitCallExpr(CE);
}

bool GlobalRWVisitor::VisitCXXConstructExpr(CXXConstructExpr *CE) {
  if (IsWritePossibleThroughFunctionParam)
    visitConstructExprArgs(CE);

  return ExecutionVisitor::VisitCXXConstructExpr(CE);
}

const llvm::SmallVector<TraversalAggregation> &
GlobalRWVisitor::getGlobalsFound() const {
  return GlobalsFound;
}

const llvm::SmallVector<ObjectTraversalAggregation> &
GlobalRWVisitor::getObjectGlobalsFound() const {
  return ObjectGlobalsFound;
}

void GlobalRWVisitor::addGlobal(DeclarationName Name, SourceLocation Loc,
                                bool IsWrite) {
  AccessKind Access = IsWrite ? AkWrite : AkRead;

  for (TraversalAggregation &Global : GlobalsFound) {
    if (Global.getDeclName() == Name) {
      Global.addGlobalRW(Loc, Access, TraversalIndex);
      return;
    }
  }

  GlobalsFound.emplace_back(Name, Loc, Access, TraversalIndex);
}

void GlobalRWVisitor::addField(DeclarationName Name,
                               FieldIndexArray FieldIndices, SourceLocation Loc,
                               bool IsWrite) {
  AccessKind Access = IsWrite ? AkWrite : AkRead;

  for (ObjectTraversalAggregation &ObjectGlobal : ObjectGlobalsFound) {
    if (ObjectGlobal.getDeclName() == Name) {
      ObjectGlobal.addFieldRW(Loc, FieldIndices, Access, TraversalIndex);
      return;
    }
  }

  ObjectGlobalsFound.emplace_back(Name, Loc, FieldIndices, Access,
                                  TraversalIndex);
}

static TraversalResultKind akToTr(AccessKind Ak) { return 1 << Ak; }

TraversalAggregation::TraversalAggregation() {}

TraversalAggregation::TraversalAggregation(DeclarationName Name,
                                           SourceLocation Loc,
                                           AccessKind Access, int Index)
    : DeclName(Name), MainPart(Index, Loc, Access) {}

void TraversalAggregation::addGlobalRW(SourceLocation Loc, AccessKind Access,
                                       int Index) {
  if (!isValid()) {
    MainPart = TraversalResult(Index, Loc, Access);
    return;
  }

  if (MainPart.IndexCreated == Index) {
    MainPart.addNewAccess(Loc, Access);
    return;
  }

  if (!hasTwoAccesses()) {
    OtherPart = TraversalResult(Index, Loc, Access);
    return;
  }

  if (OtherPart.IndexCreated == Index) {
    OtherPart.addNewAccess(Loc, Access);
    return;
  }

  switch (Access) {
  case AkWrite: {
    if (OtherPart.Kind & (TrRead | TrWrite))
      MainPart = OtherPart;

    OtherPart = TraversalResult(Index, Loc, Access);
    break;
  }
  case AkRead: {
    if (!(MainPart.Kind & TrWrite) && (OtherPart.Kind & TrWrite))
      MainPart = OtherPart;
    OtherPart = TraversalResult(Index, Loc, Access);
    break;
  }
  default: {
    break;
  }
  }
}

bool TraversalAggregation::isValid() const {
  return MainPart.Kind != TrInvalid;
}

DeclarationName TraversalAggregation::getDeclName() const { return DeclName; }

bool TraversalAggregation::hasTwoAccesses() const {
  return OtherPart.Kind != TrInvalid;
}

bool TraversalAggregation::hasConflictingOperations() const {
  return hasTwoAccesses() && ((MainPart.Kind | OtherPart.Kind) & TrWrite);
}

bool TraversalAggregation::shouldBeReported() const {
  return hasConflictingOperations();
}

TraversalResult::TraversalResult() : IndexCreated(-1), Kind(TrInvalid) {}

TraversalResult::TraversalResult(int Index, SourceLocation Location,
                                 AccessKind Access)
    : IndexCreated(Index), Kind(akToTr(Access)) {
  Loc[Access] = Location;
}

void TraversalResult::addNewAccess(SourceLocation NewLoc, AccessKind Access) {
  Kind |= 1 << Access;
  Loc[Access] = NewLoc;
}

ObjectTraversalAggregation::ObjectTraversalAggregation(
    DeclarationName Name, SourceLocation Loc, FieldIndexArray FieldIndices,
    AccessKind Access, int Index)
    : DeclName(Name), AccessTree(TraversalAggregation()) {
  addFieldRW(Loc, FieldIndices, Access, Index);
}

void ObjectTraversalAggregation::addFieldRW(SourceLocation Loc,
                                            FieldIndexArray FieldIndices,
                                            AccessKind Access, int Index) {
  ObjectAccessTree *CurrentNode = &AccessTree;
  for (FieldIndex FIndex : FieldIndices) {
    bool IsUnion = (FIndex & FiUnion) != 0;
    uint16_t FieldKey = FIndex & ~FiUnion;

    ObjectAccessTree *PrevNode = CurrentNode;
    ObjectAccessTree::FieldMap::iterator It =
        CurrentNode->Fields.find(FieldKey);

    if (It == CurrentNode->Fields.end()) {
      CurrentNode =
          new ObjectAccessTree(IsUnion ? CurrentNode->UnionTemporalAccesses
                                       : CurrentNode->OwnAccesses);
      PrevNode->Fields[FieldKey] =
          std::unique_ptr<ObjectAccessTree>(CurrentNode);
    } else {
      CurrentNode = It->second.get();
    }

    if (IsUnion) {
      if (!PrevNode->IsUnion) {
        PrevNode->IsUnion = IsUnion; // Setting the parent of the
                                     // field instead of the field
                                     // itself.
        PrevNode->UnionTemporalAccesses = PrevNode->OwnAccesses;
      }
      PrevNode->addFieldToAllExcept(FieldKey, Loc, Access, Index);
    }
  }
  CurrentNode->addFieldToAll(Loc, Access, Index);
}

bool ObjectTraversalAggregation::shouldBeReported() const {
  return shouldBeReportedRec(&AccessTree);
}

bool ObjectTraversalAggregation::shouldBeReportedRec(
    const ObjectAccessTree *Node) const {
  if (Node->OwnAccesses.hasConflictingOperations())
    return true;

  ObjectAccessTree::FieldMap::const_iterator FieldIt = Node->Fields.begin();
  ObjectAccessTree::FieldMap::const_iterator FieldsEnd = Node->Fields.end();
  for (; FieldIt != FieldsEnd; FieldIt++)
    if (shouldBeReportedRec(FieldIt->second.get()))
      return true;

  return false;
}

DeclarationName ObjectTraversalAggregation::getDeclName() const {
  return DeclName;
}

ObjectAccessTree::ObjectAccessTree(TraversalAggregation Own)
    : OwnAccesses(Own), IsUnion(false) {}

void ObjectAccessTree::addFieldToAll(SourceLocation Loc, AccessKind Access,
                                     int Index) {
  OwnAccesses.addGlobalRW(Loc, Access, Index);
  UnionTemporalAccesses.addGlobalRW(Loc, Access, Index);

  FieldMap::iterator FieldIt = Fields.begin();
  FieldMap::iterator FieldsEnd = Fields.end();

  for (; FieldIt != FieldsEnd; FieldIt++)
    FieldIt->second->addFieldToAll(Loc, Access, Index);
}

void ObjectAccessTree::addFieldToAllExcept(uint16_t ExceptIndex,
                                           SourceLocation Loc,
                                           AccessKind Access, int Index) {

  UnionTemporalAccesses.addGlobalRW(Loc, Access, Index);

  FieldMap::const_iterator FieldIt = Fields.begin();
  FieldMap::iterator FieldsEnd = Fields.end();

  for (; FieldIt != FieldsEnd; FieldIt++)
    if (FieldIt->first != ExceptIndex)
      FieldIt->second->addFieldToAll(Loc, Access, Index);
}

} // namespace clang::tidy::bugprone
