#include "clang/Analysis/Analyses/DanglingReference.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <sstream>
#include <stdbool.h>
#include <string>

namespace clang {
namespace {

template <typename T> static bool isRecordWithAttr(QualType Type) {
  auto *RD = Type->getAsCXXRecordDecl();
  if (!RD)
    return false;
  bool Result = RD->hasAttr<T>();

  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
    Result |= CTSD->getSpecializedTemplate()->getTemplatedDecl()->hasAttr<T>();
  // TODO: Should we consider attribute from base class ?
  if (RD->hasDefinition())
    for (auto B : RD->bases())
      Result |= isRecordWithAttr<T>(B.getType());
  return Result;
}

bool isOwner(QualType Q) { return isRecordWithAttr<OwnerAttr>(Q); }
bool isOwner(const Expr *E) { return isOwner(E->getType()); }
bool isOwner(const Decl *D) {
  return isa<ValueDecl>(D) &&
         isRecordWithAttr<OwnerAttr>(dyn_cast<ValueDecl>(D)->getType());
}
bool isPointer(QualType Q) {
  return Q->isNullPtrType() || Q->isPointerType() ||
         isRecordWithAttr<PointerAttr>(Q);
}
bool isPointer(const Expr *E) { return isPointer(E->getType()); }
bool isPointer(const Decl *D) {
  auto *VD = dyn_cast<ValueDecl>(D);
  return VD && isPointer(VD->getType());
}

static bool isInStlNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext();
  if (!DC)
    return false;
  if (const auto *ND = dyn_cast<NamespaceDecl>(DC))
    if (const IdentifierInfo *II = ND->getIdentifier()) {
      StringRef Name = II->getName();
      if (Name.size() >= 2 && Name.front() == '_' &&
          (Name[1] == '_' || isUppercase(Name[1])))
        return true;
    }

  return DC->isStdNamespace();
}

static bool shouldTrackImplicitObjectArg(const CXXMethodDecl *Callee) {
  if (!Callee)
    return false;
  if (auto *Conv = dyn_cast_or_null<CXXConversionDecl>(Callee))
    if (isRecordWithAttr<PointerAttr>(Conv->getConversionType()) &&
        Callee->getParent()->hasAttr<OwnerAttr>())
      return true;
  if (!isInStlNamespace(Callee->getParent()))
    return false;
  if (!isRecordWithAttr<PointerAttr>(
          Callee->getFunctionObjectParameterType()) &&
      !isRecordWithAttr<OwnerAttr>(Callee->getFunctionObjectParameterType()))
    return false;
  if (isPointer(Callee->getReturnType())) {
    if (!Callee->getIdentifier())
      return false;
    return llvm::StringSwitch<bool>(Callee->getName())
        .Cases("begin", "rbegin", "cbegin", "crbegin", true)
        .Cases("end", "rend", "cend", "crend", true)
        .Cases("c_str", "data", "get", true)
        // Map and set types.
        .Cases("find", "equal_range", "lower_bound", "upper_bound", true)
        .Default(false);
  }
  if (Callee->getReturnType()->isReferenceType()) {
    if (!Callee->getIdentifier()) {
      auto OO = Callee->getOverloadedOperator();
      if (!Callee->getParent()->hasAttr<OwnerAttr>())
        return false;
      return OO == OverloadedOperatorKind::OO_Subscript ||
             OO == OverloadedOperatorKind::OO_Star;
    }
    return llvm::StringSwitch<bool>(Callee->getName())
        .Cases("front", "back", "at", "top", "value", true)
        .Default(false);
  }
  return false;
}

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD) {
  const TypeSourceInfo *TSI = FD->getTypeSourceInfo();
  if (!TSI)
    return false;
  AttributedTypeLoc ATL;
  for (TypeLoc TL = TSI->getTypeLoc();
       (ATL = TL.getAsAdjusted<AttributedTypeLoc>());
       TL = ATL.getModifiedLoc()) {
    if (ATL.getAttrAs<LifetimeBoundAttr>())
      return true;
  }
  return false;
}

// Returns true if the given Record decl is a form of `GSLOwner<Pointer>`
// type, e.g. std::vector<string_view>, std::optional<string_view>.
static bool isContainerOfPointer(const RecordDecl *Container) {
  if (const auto *CTSD =
          dyn_cast_if_present<ClassTemplateSpecializationDecl>(Container)) {
    if (!CTSD->hasAttr<OwnerAttr>()) // Container must be a GSL owner type.
      return false;
    const auto &TAs = CTSD->getTemplateArgs();
    return TAs.size() > 0 && TAs[0].getKind() == TemplateArgument::Type &&
           isPointer(TAs[0].getAsType());
  }
  return false;
}
static bool isContainerOfOwner(const RecordDecl *Container) {
  const auto *CTSD =
      dyn_cast_if_present<ClassTemplateSpecializationDecl>(Container);
  if (!CTSD)
    return false;
  if (!CTSD->hasAttr<OwnerAttr>()) // Container must be a GSL owner type.
    return false;
  const auto &TAs = CTSD->getTemplateArgs();
  return TAs.size() > 0 && TAs[0].getKind() == TemplateArgument::Type &&
         isRecordWithAttr<OwnerAttr>(TAs[0].getAsType());
}

// Returns true if the given Record is `std::initializer_list<pointer>`.
static bool isStdInitializerListOfPointer(const RecordDecl *RD) {
  if (const auto *CTSD =
          dyn_cast_if_present<ClassTemplateSpecializationDecl>(RD)) {
    const auto &TAs = CTSD->getTemplateArgs();
    return isInStlNamespace(RD) && RD->getIdentifier() &&
           RD->getName() == "initializer_list" && TAs.size() > 0 &&
           TAs[0].getKind() == TemplateArgument::Type &&
           isPointer(TAs[0].getAsType());
  }
  return false;
}

// Returns true if the given constructor is a copy-like constructor, such as
// `Ctor(Owner<U>&&)` or `Ctor(const Owner<U>&)`.
static bool isCopyLikeConstructor(const CXXConstructorDecl *Ctor) {
  if (!Ctor || Ctor->param_size() != 1)
    return false;
  const auto *ParamRefType =
      Ctor->getParamDecl(0)->getType()->getAs<ReferenceType>();
  if (!ParamRefType)
    return false;

  // Check if the first parameter type is "Owner<U>".
  if (const auto *TST =
          ParamRefType->getPointeeType()->getAs<TemplateSpecializationType>())
    return TST->getTemplateName()
        .getAsTemplateDecl()
        ->getTemplatedDecl()
        ->hasAttr<OwnerAttr>();
  return false;
}

// Returns true if we should perform the GSL analysis on the first argument for
// the given constructor.
static bool
shouldTrackFirstArgumentForConstructor(const CXXConstructExpr *Ctor) {
  if (Ctor->getNumArgs() == 0)
    return false;
  const auto *LHSRD = Ctor->getConstructor()->getParent();
  auto RHSArgType = Ctor->getArg(0)->getType();

  // Case 1, construct a GSL pointer, e.g. std::string_view
  // Always inspect when LHS is a pointer.
  if (LHSRD->hasAttr<PointerAttr>())
    return isPointer(RHSArgType) || isOwner(RHSArgType);
  if (Ctor->getConstructor()->param_empty() || !isContainerOfPointer(LHSRD))
    return false;

  // Now, the LHS is an Owner<Pointer> type, e.g., std::vector<string_view>.
  //
  // At a high level, we cannot precisely determine what the nested pointer
  // owns. However, by analyzing the RHS owner type, we can use heuristics to
  // infer ownership information. These heuristics are designed to be
  // conservative, minimizing false positives while still providing meaningful
  // diagnostics.
  //
  // While this inference isn't perfect, it helps catch common use-after-free
  // patterns.
  const auto *RHSRD = RHSArgType->getAsRecordDecl();
  // LHS is constructed from an intializer_list.
  //
  // std::initializer_list is a proxy object that provides access to the backing
  // array. We perform analysis on it to determine if there are any dangling
  // temporaries in the backing array.
  // E.g. std::vector<string_view> abc = {string()};
  if (isStdInitializerListOfPointer(RHSRD))
    return true;

  // RHS can be a Pointer.
  if (isPointer(RHSArgType))
    return true;

  // RHS must be an owner.
  if (!isOwner(RHSArgType))
    return false;

  // Bail out if the RHS is Owner<Pointer>.
  //
  // We cannot reliably determine what the LHS nested pointer owns -- it could
  // be the entire RHS or the nested pointer in RHS. To avoid false positives,
  // we skip this case, such as:
  //   std::stack<std::string_view> s(std::deque<std::string_view>{});
  //
  // TODO: this also has a false negative, it doesn't catch the case like:
  //   std::optional<span<int*>> os = std::vector<int*>{}
  if (isContainerOfPointer(RHSRD) && LHSRD != RHSRD)
    return false;

  // Assume that the nested Pointer is constructed from the nested Owner.
  // E.g. std::optional<string_view> sv = std::optional<string>(s);
  if (isContainerOfOwner(RHSRD))
    return true;

  // Now, the LHS is an Owner<Pointer> and the RHS is an Owner<X>,  where X is
  // neither an `Owner` nor a `Pointer`.
  //
  // Use the constructor's signature as a hint. If it is a copy-like constructor
  // `Owner1<Pointer>(Owner2<X>&&)`, we assume that the nested pointer is
  // constructed from X. In such cases, we do not diagnose, as `X` is not an
  // owner, e.g.
  //   std::optional<string_view> sv = std::optional<Foo>();
  if (const auto *PrimaryCtorTemplate =
          Ctor->getConstructor()->getPrimaryTemplate();
      PrimaryCtorTemplate &&
      isCopyLikeConstructor(dyn_cast_if_present<CXXConstructorDecl>(
          PrimaryCtorTemplate->getTemplatedDecl()))) {
    return false;
  }
  // Assume that the nested pointer is constructed from the whole RHS.
  // E.g. optional<string_view> s = std::string();
  return true;
}

bool shouldTrackAsPointer(QualType QT) {
  if (isPointer(QT))
    return true;
  if (auto *RD = QT->getAsRecordDecl())
    return isContainerOfPointer(RD);
  return false;
}

bool shouldTrackAsPointer(const Decl *D) {
  auto *VD = dyn_cast<VarDecl>(D);
  if (!VD)
    return false;
  return shouldTrackAsPointer(VD->getType());
}

bool doesParmPointsToArgument(const ParmVarDecl *PVD) {
  // Prefer only pointer args. Skip containers.
  // Default args live as long as the function call expr. Ignore.
  return isRecordWithAttr<PointerAttr>(PVD->getType()) && !PVD->hasDefaultArg();
}

struct MemoryLoc {
  enum Storage {
    EMPTY,            // Pointer is null.
    STACK,            // Pointer points to something on stack.
    UNKNOWN_ARGUMENT, // Pointer points to an argument provided to the function.
    UNKNOWN,          // Pointer points to an unknown entity.
  } Loc;
  // Details of stack location.
  const Decl *D = nullptr;
  const Expr *MTE = nullptr;

  bool IsEmpty() { return Loc == EMPTY; }
  bool IsOnStack() { return Loc == STACK; }
  bool IsArgument() { return Loc == UNKNOWN_ARGUMENT; }
  bool IsUnknown() { return Loc == UNKNOWN || IsArgument(); }

  const Decl *getDecl() { return D; }
  const Expr *getExpr() { return MTE; }

  static MemoryLoc Argument(const Decl *D) {
    return {UNKNOWN_ARGUMENT, D, nullptr};
  }
  static MemoryLoc Unknown() { return {UNKNOWN, nullptr, nullptr}; }
  static MemoryLoc Empty() { return {EMPTY, nullptr, nullptr}; }
  static MemoryLoc VarOnStack(const Decl *D) { return {STACK, D, nullptr}; }
  static MemoryLoc Temporary(const Expr *MTE) { return {STACK, nullptr, MTE}; }

  std::string str() const {
    std::ostringstream os;
    switch (Loc) {
    case EMPTY:
      os << "Empty";
      break;
    case UNKNOWN_ARGUMENT:
      os << "Argument(unknown)";
      if (auto *VD = dyn_cast_or_null<VarDecl>(D))
        os << " \"" << VD->getName().str() << "\"";
      break;
    case UNKNOWN:
      os << "Unknown";
      break;
    case STACK:
      os << "Stack";
      if (auto *VD = dyn_cast_or_null<VarDecl>(D))
        os << " \"" << VD->getName().str() << "\"";
      if (MTE)
        os << " (temporary)";
      break;
    }
    return os.str();
  }

  static inline void Profile(llvm::FoldingSetNodeID &ID, const MemoryLoc &M) {
    ID.AddInteger(M.Loc);
    ID.AddPointer(M.D);
    ID.AddPointer(M.MTE);
  }

  inline void Profile(llvm::FoldingSetNodeID &ID) const {
    return Profile(ID, *this);
  }
};
inline bool operator==(const MemoryLoc &LHS, const MemoryLoc &RHS) {
  return LHS.Loc == RHS.Loc && LHS.D == RHS.D && LHS.MTE == RHS.MTE;
}
inline bool operator!=(const MemoryLoc &LHS, const MemoryLoc &RHS) {
  return !(LHS == RHS);
}

class PointsToFactory {
public:
  PointsToFactory()
      : ESetFact(false), DSetFact(false), EPointsToFact(false),
        DPointsToFact(false) {}
  llvm::ImmutableSet<const Expr *>::Factory ESetFact;
  llvm::ImmutableSet<const Decl *>::Factory DSetFact;
  llvm::ImmutableMap<const Expr *, MemoryLoc>::Factory EPointsToFact;
  llvm::ImmutableMap<const Decl *, MemoryLoc>::Factory DPointsToFact;
};

class PointsToSet {

public:
  explicit PointsToSet(PointsToFactory *Factory)
      : Factory(Factory), ExprPointsTo(Factory->EPointsToFact.getEmptyMap()),
        DeclPointsTo(Factory->DPointsToFact.getEmptyMap()),
        StackExprs(Factory->ESetFact.getEmptySet()),
        StackDecls(Factory->DSetFact.getEmptySet()),
        UnknownDecls(Factory->DSetFact.getEmptySet()) {}

  PointsToSet(const PointsToSet &P) = default;

  PointsToSet &operator=(const PointsToSet &P) = default;

  bool ContainsExprPointsTo(const Expr *E) const {
    return E && ExprPointsTo.contains(E);
  }
  bool ContainsDeclPointsTo(const Decl *D) const {
    return D && DeclPointsTo.contains(D);
  }
  MemoryLoc GetExprPointsTo(const Expr *E) const {
    assert(ContainsExprPointsTo(E));
    return *ExprPointsTo.lookup(E);
  }
  MemoryLoc GetDeclPointsTo(const Decl *D) const {
    assert(ContainsDeclPointsTo(D));
    return *DeclPointsTo.lookup(D);
  }
  MemoryLoc SetExprPointer(const Expr *E, MemoryLoc M) {
    ExprPointsTo = Factory->EPointsToFact.add(ExprPointsTo, E, M);
    return M;
  }
  MemoryLoc SetDeclPointer(const Decl *D, MemoryLoc M) {
    if (DeclPointsTo.contains(D))
      DeclPointsTo = Factory->DPointsToFact.remove(DeclPointsTo, D);
    DeclPointsTo = Factory->DPointsToFact.add(DeclPointsTo, D, M);
    return M;
  }

  void AddToStack(const Decl *D) {
    StackDecls = Factory->DSetFact.add(StackDecls, D);
  }
  void MarkAsUnknown(const Decl *D) {
    UnknownDecls = Factory->DSetFact.add(UnknownDecls, D);
  }
  void AddToStack(const Expr *E) {
    StackExprs = Factory->ESetFact.add(StackExprs, E);
  }

  bool IsOnStack(const Decl *D) const {
    return StackDecls.contains(D) && !UnknownDecls.contains(D);
  }
  bool IsOnStack(const Expr *E) const { return StackExprs.contains(E); }

  int size() const {
    return ExprPointsTo.getHeight() + DeclPointsTo.getHeight() +
           StackExprs.getHeight() + StackDecls.getHeight() +
           UnknownDecls.getHeight();
  }
  void dump() {
    std::ostringstream os;
    for (const auto &x : DeclPointsTo) {
      auto *VD = dyn_cast<VarDecl>(x.first);
      if (VD) {
        llvm::errs() << VD->getNameAsString() << " ----> Points to ---> "
                     << x.second.str() << "\n";
      }
    }
    for (const auto &x : UnknownDecls) {
      auto *VD = dyn_cast<VarDecl>(x);
      if (VD) {
        llvm::errs() << "\tunknown: " << VD->getNameAsString() << "\n";
      }
    }
  }

private:
  PointsToFactory *Factory;

  // TODO: Make private.
public:
  // TODO: We only need decl information. not expr!

  // Map from an expression of View type to its pointee and Owner type to the
  // reference<TODO simplify>. This should follow single assignment because
  // Expr* cannot be reassigned in the program.
  llvm::ImmutableMap<const Expr *, MemoryLoc> ExprPointsTo;
  // Map from a decl of View type to it pointee. This can be reassigned at
  // various points in the program due to transfer functions.
  llvm::ImmutableMap<const Decl *, MemoryLoc> DeclPointsTo;

  // Collection of Expr* and Decl* stored on stack.
  llvm::ImmutableSet<const Expr *> StackExprs;
  llvm::ImmutableSet<const Decl *> StackDecls;
  // Decl was previously on stack but was then moved to some unknown storage.
  // For example: std::unique_ptr<> P can be moved to unknown storage using
  // std::move(P) or P.release();
  llvm::ImmutableSet<const Decl *> UnknownDecls;
};

PointsToSet Merge(const PointsToSet a, const PointsToSet b) {
  if (a.size() < b.size())
    return Merge(b, a);
  PointsToSet res = a;
  for (const auto &v : b.ExprPointsTo) {
    if (!res.ContainsExprPointsTo(v.first))
      res.SetExprPointer(v.first, v.second);
    else if (res.GetExprPointsTo(v.first) != v.second)
      res.SetExprPointer(v.first, MemoryLoc::Unknown());
  }
  for (const auto &v : b.DeclPointsTo) {
    if (!res.ContainsDeclPointsTo(v.first))
      res.SetDeclPointer(v.first, v.second);
    else if (res.GetDeclPointsTo(v.first) != v.second)
      res.SetDeclPointer(v.first, MemoryLoc::Unknown());
  }
  for (const auto *D : b.StackDecls) {
    res.AddToStack(D);
  }
  for (const auto *E : b.StackExprs) {
    res.AddToStack(E);
  }
  for (const auto *D : b.UnknownDecls) {
    res.MarkAsUnknown(D);
  }
  return res;
}

class BlockVisitor : public ConstStmtVisitor<BlockVisitor> {
public:
  BlockVisitor(const clang::CFGBlock *B, PointsToSet Incoming,
               const DeclContext &DC, LiveVariables *LV,
               DanglingReferenceReporter *Reporter)
      : B(B), PointsTo(Incoming), DC(DC), LV(LV), Reporter(Reporter) {}
  void Handle() {
    for (CFGBlock::const_iterator BI = B->begin(), BE = B->end(); BI != BE;
         ++BI) {
      // BI->dump();
      if (auto cfgstmt = BI->getAs<CFGStmt>()) {
        // llvm::errs() << "================================================\n";
        // cfgstmt->dump();
        auto *stmt = cfgstmt->getStmt();
        // if (auto *E = dyn_cast<Expr>(stmt))
        //   E->dumpColor();

        Handle(stmt);

        // if (auto *E = dyn_cast<Expr>(stmt)) {
        //   E->dumpColor();
        //   auto Loc = ResolveExpr(E);
        //   llvm::errs() << "Points To : " << Loc.str() << "\n";
        // }

        if (auto *RS = dyn_cast<ReturnStmt>(stmt))
          DiagnoseReturnStmt(RS);
      }
      if (auto dtor = BI->getAs<CFGAutomaticObjDtor>()) {
        DiagnoseDanglingReference(dtor->getVarDecl());
      }
    }
  }
  PointsToSet getOutgoing() { return PointsTo; }

private:
  void DiagnoseDanglingReference(const VarDecl *VD) {
    for (const auto &PT : PointsTo.DeclPointsTo) {
      MemoryLoc Pointee = PT.second;
      if (Pointee.getDecl() != VD)
        continue;
      const VarDecl *Pointer = dyn_cast_or_null<VarDecl>(PT.first);
      if (!Pointer)
        continue;
      if (LV && LV->isLive(B, Pointer)) {
        // We have a use-after-free!
        // TODO: Show more helpful diagnostics.
        Reporter->ReportDanglingReference(VD);
        return;
      }
    }
  }

  // Diagnose possible return of stack variable.
  void DiagnoseReturnStmt(const ReturnStmt *RS) {
    if (!RS)
      return;
    const Expr *RetExpr = RS->getRetValue();
    if (!RetExpr)
      return;
    if (!shouldTrackAsPointer(RetExpr->getType()))
      return;
    // Handle(RetExpr);
    auto RetPointee = ResolveExpr(RetExpr);
    // RetExpr->dumpColor();
    // llvm::errs() << "Returning pointer to " << RetPointee.str() << "\n";
    if (RetPointee.IsOnStack()) {
      llvm::errs() << "=================================================\n";
      llvm::errs() << "Returning pointer to " << RetPointee.str() << "\n\n";
      RetExpr->dumpColor();
      // This points to something on stack!!
      if (auto *D = RetPointee.getDecl(); D && PointsTo.IsOnStack(D))
        Reporter->ReportReturnLocalVar(RetExpr, D);
      if (auto *E = RetPointee.getExpr())
        Reporter->ReportReturnTemporaryExpr(E);
    } else if (RetPointee.IsArgument()) {
      auto *PVD = dyn_cast<ParmVarDecl>(RetPointee.getDecl());
      // TODO: This can be split in more granular diagnostics based on the type
      // of function. Eg. Visibility of a function (private member fn, static,
      // anonymous namespace Vs. public members fn, fn in headers).
      // TODO: Templates should not be suggested.
      // TODO: Add an infered annotation so that it is available atleast in the
      // same TU.
      if (!PVD->hasAttr<LifetimeBoundAttr>())
        Reporter->SuggestLifetimebound(PVD, RetExpr);
    }
  }

  void Handle(const Stmt *S) {
    if (auto *E = dyn_cast<Expr>(S); !PointsTo.ContainsExprPointsTo(E))
      Visit(S);
  }

  void MaybeInitaliseDecl(const Decl *D) {
    if (!D)
      return;
    auto *VD = dyn_cast<VarDecl>(D);
    if (!VD)
      return;
    // Initialise the pointer if we are seeing it for the first time.
    if (shouldTrackAsPointer(VD->getType())) {
      if (PointsTo.ContainsDeclPointsTo(D))
        return;
      // Pointer params are always unknown.
      if (auto *PVD = dyn_cast<ParmVarDecl>(VD)) {
        UpdatePointer(VD, doesParmPointsToArgument(PVD)
                              ? MemoryLoc::Argument(VD)
                              : MemoryLoc::Unknown());
        return;
      }
      UpdatePointer(VD, ResolveExpr(VD->getInit()));
      return;
    }
    // Ignore non-stack variable.
    // TODO: Track reference variables.
    if (!DC.containsDecl(const_cast<Decl *>(D)) || !VD->hasLocalStorage() ||
        VD->getType()->isReferenceType())
      return;

    if (!PointsTo.IsOnStack(D))
      AddToStack(VD);
  }

public:
  void VisitDeclStmt(const DeclStmt *DS) {
    MaybeInitaliseDecl(DS->getSingleDecl());
  }

  void VisitDeclRefExpr(const DeclRefExpr *DRE) {
    SetExprPointer(DRE, DRE->getDecl());
  }

  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *MTE) {
    // Ignore MTE of pointer types.
    if (shouldTrackAsPointer(MTE->getType())) {
      Handle(MTE->getSubExpr());
      SetExprPointer(MTE, MTE->getSubExpr());
      return;
    }
    // We have a temporary on stack.
    AddToStack(MTE);
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *E) {
    auto *SE = E->IgnoreImpCasts();
    Handle(SE);
    SetExprPointer(E, SE);
  }

  void VisitExprWithCleanups(const ExprWithCleanups *E) {
    SetExprPointer(E, E->getSubExpr());
  }
  bool IsUniquePtrRelease(const CXXMemberCallExpr *MCE) {
    if (!MCE || !MCE->getCalleeDecl())
      return false;
    auto *FD = MCE->getCalleeDecl()->getAsFunction();
    if (!FD)
      return false;
    return FD->getIdentifier() && FD->getName() == "release";
  }
  void VisitCallExpr(const CallExpr *CE) {
    if (CE->isCallToStdMove()) {
      assert(CE->getNumArgs() == 1);
      MarkDeclAsUnknown(CE->getArg(0));
      return;
    }
    // TODO: Can be merged with above std::move.
    if (auto *MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      if (IsUniquePtrRelease(MCE))
        MarkDeclAsUnknown(MCE->getImplicitObjectArgument());
    }
    // Lifetimebound function calls.
    auto *FD = dyn_cast_or_null<FunctionDecl>(CE->getCalleeDecl());
    if (!FD)
      return;
    // FIXME: This should only be done for GSL pointer args and not all args!
    QualType RetType = FD->getReturnType();
    if (RetType->isReferenceType() && !isOwner(RetType->getPointeeType()))
      return;
    Expr *ObjectArg = nullptr;
    if (auto *MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
      ObjectArg = MCE->getImplicitObjectArgument();
      if (ObjectArg && (implicitObjectParamIsLifetimeBound(FD) ||
                        shouldTrackImplicitObjectArg(MCE->getMethodDecl()))) {
        // FIXME: Gives a false positive:
        //        std::vector<int*> a = StatusOr<std::vector<int*>>{}.value();
        // TODO: Track more args. Not just the first one!
        SetExprPointer(CE, ObjectArg);
        return;
      }
    }
    for (unsigned I = 0; I < FD->getNumParams(); ++I) {
      const ParmVarDecl *PVD = FD->getParamDecl(I);
      if (CE->getArg(I) && PVD->hasAttr<LifetimeBoundAttr>()) {
        // TODO: Track more args. Not just the first one!
        SetExprPointer(CE, CE->getArg(I));
        return;
      }
    }
  }

  void VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
    if (shouldTrackFirstArgumentForConstructor(CCE)) {
      SetExprPointer(CCE, CCE->getArg(0));
    }
  }

  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
    if (OCE->isAssignmentOp()) {
      assert(OCE->getNumArgs() == 2);
      Handle(OCE->getArg(0));
      Handle(OCE->getArg(1));
      HandleAssignment(OCE->getArg(0), OCE->getArg(1));
    }
  }

private:
  // Returns the Decl that is aliased by this expression.
  const Decl *DeclReferencedBy(const Expr *E) {
    if (auto *DRE = dyn_cast_or_null<DeclRefExpr>(E)) {
      return DRE->getDecl();
    }
    return nullptr;
  }

  void MarkDeclAsUnknown(const Expr *E) {
    if (!E)
      return;
    if (const Decl *D = DeclReferencedBy(E)) {
      PointsTo.MarkAsUnknown(D);
    }
  }

  void HandleAssignment(const Expr *A, const Expr *B) {
    if (!shouldTrackAsPointer(A->getType()))
      return;

    if (const Decl *PointerDecl = DeclReferencedBy(A);
        PointerDecl && shouldTrackAsPointer(PointerDecl))
      UpdatePointer(PointerDecl, B);
  }

  // Update the contents of a Pointer.
  void UpdatePointer(const Decl *PointerD, MemoryLoc ML) {
    assert(shouldTrackAsPointer(PointerD));
    PointsTo.SetDeclPointer(PointerD, ML);
  }
  void UpdatePointer(const Decl *PointerD, const Expr *A) {
    UpdatePointer(PointerD, ResolveExpr(A));
  }

  void SetExprPointer(const Expr *E, MemoryLoc ML) {
    assert(!PointsTo.ContainsExprPointsTo(E));
    PointsTo.SetExprPointer(E, ML);
  }
  void SetExprPointer(const Expr *E, const Expr *PointeeE) {
    SetExprPointer(E, ResolveExpr(PointeeE));
  }
  void SetExprPointer(const Expr *E, const Decl *D) {
    SetExprPointer(E, ResolveDecl(D));
  }

  MemoryLoc ResolveExpr(const Expr *E) {
    if (!E)
      return MemoryLoc::Empty();
    if (PointsTo.ContainsExprPointsTo(E)) {
      return PointsTo.GetExprPointsTo(E);
    }
    Handle(E);
    if (PointsTo.ContainsExprPointsTo(E))
      return PointsTo.GetExprPointsTo(E);
    return PointsTo.SetExprPointer(E, MemoryLoc::Unknown());
  }

  // Returns the memory location pointed to by D. If D is a pointer-type,
  // returns the memory pointed to by the pointer.
  MemoryLoc ResolveDecl(const Decl *D) {
    MaybeInitaliseDecl(D);
    if (shouldTrackAsPointer(D))
      return ResolvePointer(D);
    return ResolveNonPointer(D);
  }

  MemoryLoc ResolvePointer(const Decl *D) {
    auto *VD = dyn_cast<VarDecl>(D);
    // TODO: Handle other decls like field.
    if (!VD || !VD->hasLocalStorage())
      return MemoryLoc::Unknown();
    assert(shouldTrackAsPointer(D));
    return PointsTo.GetDeclPointsTo(D);
  }

  MemoryLoc ResolveNonPointer(const Decl *D) {
    assert(!shouldTrackAsPointer(D));
    if (IsOnStack(D)) {
      return MemoryLoc::VarOnStack(D);
    }
    return MemoryLoc::Unknown();
  }

  void AddToStack(const Decl *D) { PointsTo.AddToStack(D); }
  void AddToStack(const Expr *E) {
    assert(isa<MaterializeTemporaryExpr>(E));
    assert(!isPointer(E));
    PointsTo.AddToStack(E);
    // Add a self edge.
    assert(!PointsTo.ContainsExprPointsTo(E));
    PointsTo.SetExprPointer(E, MemoryLoc::Temporary(E));
  }
  bool IsOnStack(const Decl *D) { return PointsTo.IsOnStack(D); }
  bool IsOnStack(const Expr *E) { return PointsTo.IsOnStack(E); }

  const clang::CFGBlock *B;
  PointsToSet PointsTo;
  const DeclContext &DC;
  LiveVariables *LV;
  DanglingReferenceReporter *Reporter;
};

class DanglingReferenceAnalyzer {
public:
  DanglingReferenceAnalyzer(const DeclContext &DC, const CFG &cfg,
                            AnalysisDeclContext &AC, LiveVariables *LV,
                            DanglingReferenceReporter *Reporter)
      : DC(DC), cfg(cfg), AC(AC), LV(LV), Reporter(Reporter) {}
  void RunAnalysis() {
    // cfg.dump(AC.getASTContext().getLangOpts(), true);
    ForwardDataflowWorklist worklist(cfg, AC);
    worklist.enqueueSuccessors(&cfg.getEntry());
    PointsToSets.insert({&cfg.getEntry(), PointsToSet(&Factory)});

    llvm::BitVector Visited(cfg.getNumBlockIDs());

    while (const CFGBlock *Block = worklist.dequeue()) {
      unsigned BlockID = Block->getBlockID();
      if (Visited[BlockID])
        continue;
      // llvm::errs() << "====================================\n";
      // Block->dump();
      PointsToSet IncomingPointsTo = MergePredecessors(Block);
      BlockVisitor BV(Block, IncomingPointsTo, DC, LV, Reporter);
      BV.Handle();
      PointsToSets.insert({Block, BV.getOutgoing()});

      worklist.enqueueSuccessors(Block);
      Visited[BlockID] = true;
    }
  }

  PointsToSet MergePredecessors(const CFGBlock *Block) {
    PointsToSet Result(&Factory);
    for (const auto &Pred : Block->preds())
      if (PointsToSets.contains(Pred))
        Result = Merge(Result, PointsToSets.find(Pred)->getSecond());
    return Result;
  }

  PointsToFactory Factory;
  [[maybe_unused]] const DeclContext &DC;
  const CFG &cfg;
  AnalysisDeclContext &AC;
  LiveVariables *LV;
  DanglingReferenceReporter *Reporter;
  llvm::DenseMap<const CFGBlock *, PointsToSet> PointsToSets;
};
} // namespace

void runDanglingReferenceAnalysis(const DeclContext &DC, const CFG &cfg,
                                  AnalysisDeclContext &AC,
                                  DanglingReferenceReporter *Reporter) {
  std::unique_ptr<LiveVariables> LV =
      LiveVariables::computeLiveness(AC, /*killAtAssign=*/false);
  DanglingReferenceAnalyzer DRA(DC, cfg, AC, LV.get(), Reporter);
  DRA.RunAnalysis();
}

} // namespace clang
