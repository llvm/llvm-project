//===- LifetimeSafety.cpp - C++ Lifetime Safety Analysis -*--------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/LifetimeSafety.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <cstdint>
#include <memory>
#include <optional>

namespace clang::lifetimes {
namespace internal {

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable.
/// TODO: Model access paths of other types, e.g., s.field, heap and globals.
struct AccessPath {
  const clang::ValueDecl *D;

  AccessPath(const clang::ValueDecl *D) : D(D) {}
};

/// Information about a single borrow, or "Loan". A loan is created when a
/// reference or pointer is created.
struct Loan {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
  LoanID ID;
  AccessPath Path;
  /// The expression that creates the loan, e.g., &x.
  const Expr *IssueExpr;

  Loan(LoanID id, AccessPath path, const Expr *IssueExpr)
      : ID(id), Path(path), IssueExpr(IssueExpr) {}

  void dump(llvm::raw_ostream &OS) const {
    OS << ID << " (Path: ";
    OS << Path.D->getNameAsString() << ")";
  }
};

/// An Origin is a symbolic identifier that represents the set of possible
/// loans a pointer-like object could hold at any given time.
/// TODO: Enhance the origin model to handle complex types, pointer
/// indirection and reborrowing. The plan is to move from a single origin per
/// variable/expression to a "list of origins" governed by the Type.
/// For example, the type 'int**' would have two origins.
/// See discussion:
/// https://github.com/llvm/llvm-project/pull/142313/commits/0cd187b01e61b200d92ca0b640789c1586075142#r2137644238
struct Origin {
  OriginID ID;
  /// A pointer to the AST node that this origin represents. This union
  /// distinguishes between origins from declarations (variables or parameters)
  /// and origins from expressions.
  llvm::PointerUnion<const clang::ValueDecl *, const clang::Expr *> Ptr;

  Origin(OriginID ID, const clang::ValueDecl *D) : ID(ID), Ptr(D) {}
  Origin(OriginID ID, const clang::Expr *E) : ID(ID), Ptr(E) {}

  const clang::ValueDecl *getDecl() const {
    return Ptr.dyn_cast<const clang::ValueDecl *>();
  }
  const clang::Expr *getExpr() const {
    return Ptr.dyn_cast<const clang::Expr *>();
  }
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
public:
  LoanManager() = default;

  Loan &addLoan(AccessPath Path, const Expr *IssueExpr) {
    AllLoans.emplace_back(getNextLoanID(), Path, IssueExpr);
    return AllLoans.back();
  }

  const Loan &getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }
  llvm::ArrayRef<Loan> getLoans() const { return AllLoans; }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<Loan> AllLoans;
};

/// Manages the creation, storage, and retrieval of origins for pointer-like
/// variables and expressions.
class OriginManager {
public:
  OriginManager() = default;

  Origin &addOrigin(OriginID ID, const clang::ValueDecl &D) {
    AllOrigins.emplace_back(ID, &D);
    return AllOrigins.back();
  }
  Origin &addOrigin(OriginID ID, const clang::Expr &E) {
    AllOrigins.emplace_back(ID, &E);
    return AllOrigins.back();
  }

  // TODO: Mark this method as const once we remove the call to getOrCreate.
  OriginID get(const Expr &E) {
    auto It = ExprToOriginID.find(&E);
    if (It != ExprToOriginID.end())
      return It->second;
    // If the expression itself has no specific origin, and it's a reference
    // to a declaration, its origin is that of the declaration it refers to.
    // For pointer types, where we don't pre-emptively create an origin for the
    // DeclRefExpr itself.
    if (const auto *DRE = dyn_cast<DeclRefExpr>(&E))
      return get(*DRE->getDecl());
    // TODO: This should be an assert(It != ExprToOriginID.end()). The current
    // implementation falls back to getOrCreate to avoid crashing on
    // yet-unhandled pointer expressions, creating an empty origin for them.
    return getOrCreate(E);
  }

  OriginID get(const ValueDecl &D) {
    auto It = DeclToOriginID.find(&D);
    // TODO: This should be an assert(It != DeclToOriginID.end()). The current
    // implementation falls back to getOrCreate to avoid crashing on
    // yet-unhandled pointer expressions, creating an empty origin for them.
    if (It == DeclToOriginID.end())
      return getOrCreate(D);

    return It->second;
  }

  OriginID getOrCreate(const Expr &E) {
    auto It = ExprToOriginID.find(&E);
    if (It != ExprToOriginID.end())
      return It->second;

    OriginID NewID = getNextOriginID();
    addOrigin(NewID, E);
    ExprToOriginID[&E] = NewID;
    return NewID;
  }

  const Origin &getOrigin(OriginID ID) const {
    assert(ID.Value < AllOrigins.size());
    return AllOrigins[ID.Value];
  }

  llvm::ArrayRef<Origin> getOrigins() const { return AllOrigins; }

  OriginID getOrCreate(const ValueDecl &D) {
    auto It = DeclToOriginID.find(&D);
    if (It != DeclToOriginID.end())
      return It->second;
    OriginID NewID = getNextOriginID();
    addOrigin(NewID, D);
    DeclToOriginID[&D] = NewID;
    return NewID;
  }

  void dump(OriginID OID, llvm::raw_ostream &OS) const {
    OS << OID << " (";
    Origin O = getOrigin(OID);
    if (const ValueDecl *VD = O.getDecl())
      OS << "Decl: " << VD->getNameAsString();
    else if (const Expr *E = O.getExpr())
      OS << "Expr: " << E->getStmtClassName();
    else
      OS << "Unknown";
    OS << ")";
  }

private:
  OriginID getNextOriginID() { return NextOriginID++; }

  OriginID NextOriginID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<Origin> AllOrigins;
  llvm::DenseMap<const clang::ValueDecl *, OriginID> DeclToOriginID;
  llvm::DenseMap<const clang::Expr *, OriginID> ExprToOriginID;
};

/// An abstract base class for a single, atomic lifetime-relevant event.
class Fact {

public:
  enum class Kind : uint8_t {
    /// A new loan is issued from a borrow expression (e.g., &x).
    Issue,
    /// A loan expires as its underlying storage is freed (e.g., variable goes
    /// out of scope).
    Expire,
    /// An origin is propagated from a source to a destination (e.g., p = q).
    /// This can also optionally kill the destination origin before flowing into
    /// it. Otherwise, the source's loan set is merged into the destination's
    /// loan set.
    OriginFlow,
    /// An origin escapes the function by flowing into the return value.
    ReturnOfOrigin,
    /// An origin is used (eg. appears as l-value expression like DeclRefExpr).
    Use,
    /// A marker for a specific point in the code, for testing.
    TestPoint,
  };

private:
  Kind K;

protected:
  Fact(Kind K) : K(K) {}

public:
  virtual ~Fact() = default;
  Kind getKind() const { return K; }

  template <typename T> const T *getAs() const {
    if (T::classof(this))
      return static_cast<const T *>(this);
    return nullptr;
  }

  virtual void dump(llvm::raw_ostream &OS, const LoanManager &,
                    const OriginManager &) const {
    OS << "Fact (Kind: " << static_cast<int>(K) << ")\n";
  }
};

class IssueFact : public Fact {
  LoanID LID;
  OriginID OID;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Issue; }

  IssueFact(LoanID LID, OriginID OID) : Fact(Kind::Issue), LID(LID), OID(OID) {}
  LoanID getLoanID() const { return LID; }
  OriginID getOriginID() const { return OID; }
  void dump(llvm::raw_ostream &OS, const LoanManager &LM,
            const OriginManager &OM) const override {
    OS << "Issue (";
    LM.getLoan(getLoanID()).dump(OS);
    OS << ", ToOrigin: ";
    OM.dump(getOriginID(), OS);
    OS << ")\n";
  }
};

class ExpireFact : public Fact {
  LoanID LID;
  SourceLocation ExpiryLoc;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Expire; }

  ExpireFact(LoanID LID, SourceLocation ExpiryLoc)
      : Fact(Kind::Expire), LID(LID), ExpiryLoc(ExpiryLoc) {}

  LoanID getLoanID() const { return LID; }
  SourceLocation getExpiryLoc() const { return ExpiryLoc; }

  void dump(llvm::raw_ostream &OS, const LoanManager &LM,
            const OriginManager &) const override {
    OS << "Expire (";
    LM.getLoan(getLoanID()).dump(OS);
    OS << ")\n";
  }
};

class OriginFlowFact : public Fact {
  OriginID OIDDest;
  OriginID OIDSrc;
  // True if the destination origin should be killed (i.e., its current loans
  // cleared) before the source origin's loans are flowed into it.
  bool KillDest;

public:
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::OriginFlow;
  }

  OriginFlowFact(OriginID OIDDest, OriginID OIDSrc, bool KillDest)
      : Fact(Kind::OriginFlow), OIDDest(OIDDest), OIDSrc(OIDSrc),
        KillDest(KillDest) {}

  OriginID getDestOriginID() const { return OIDDest; }
  OriginID getSrcOriginID() const { return OIDSrc; }
  bool getKillDest() const { return KillDest; }

  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override {
    OS << "OriginFlow (Dest: ";
    OM.dump(getDestOriginID(), OS);
    OS << ", Src: ";
    OM.dump(getSrcOriginID(), OS);
    OS << (getKillDest() ? "" : ", Merge");
    OS << ")\n";
  }
};

class ReturnOfOriginFact : public Fact {
  OriginID OID;

public:
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::ReturnOfOrigin;
  }

  ReturnOfOriginFact(OriginID OID) : Fact(Kind::ReturnOfOrigin), OID(OID) {}
  OriginID getReturnedOriginID() const { return OID; }
  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override {
    OS << "ReturnOfOrigin (";
    OM.dump(getReturnedOriginID(), OS);
    OS << ")\n";
  }
};

class UseFact : public Fact {
  const Expr *UseExpr;
  // True if this use is a write operation (e.g., left-hand side of assignment).
  // Write operations are exempted from use-after-free checks.
  bool IsWritten = false;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Use; }

  UseFact(const Expr *UseExpr) : Fact(Kind::Use), UseExpr(UseExpr) {}

  OriginID getUsedOrigin(const OriginManager &OM) const {
    // TODO: Remove const cast and make OriginManager::get as const.
    return const_cast<OriginManager &>(OM).get(*UseExpr);
  }
  const Expr *getUseExpr() const { return UseExpr; }
  void markAsWritten() { IsWritten = true; }
  bool isWritten() const { return IsWritten; }

  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override {
    OS << "Use (";
    OM.dump(getUsedOrigin(OM), OS);
    OS << ", " << (isWritten() ? "Write" : "Read") << ")\n";
  }
};

/// A dummy-fact used to mark a specific point in the code for testing.
/// It is generated by recognizing a `void("__lifetime_test_point_...")` cast.
class TestPointFact : public Fact {
  StringRef Annotation;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::TestPoint; }

  explicit TestPointFact(StringRef Annotation)
      : Fact(Kind::TestPoint), Annotation(Annotation) {}

  StringRef getAnnotation() const { return Annotation; }

  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &) const override {
    OS << "TestPoint (Annotation: \"" << getAnnotation() << "\")\n";
  }
};

class FactManager {
public:
  llvm::ArrayRef<const Fact *> getFacts(const CFGBlock *B) const {
    auto It = BlockToFactsMap.find(B);
    if (It != BlockToFactsMap.end())
      return It->second;
    return {};
  }

  void addBlockFacts(const CFGBlock *B, llvm::ArrayRef<Fact *> NewFacts) {
    if (!NewFacts.empty())
      BlockToFactsMap[B].assign(NewFacts.begin(), NewFacts.end());
  }

  template <typename FactType, typename... Args>
  FactType *createFact(Args &&...args) {
    void *Mem = FactAllocator.Allocate<FactType>();
    return new (Mem) FactType(std::forward<Args>(args)...);
  }

  void dump(const CFG &Cfg, AnalysisDeclContext &AC) const {
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << "       Lifetime Analysis Facts:\n";
    llvm::dbgs() << "==========================================\n";
    if (const Decl *D = AC.getDecl())
      if (const auto *ND = dyn_cast<NamedDecl>(D))
        llvm::dbgs() << "Function: " << ND->getQualifiedNameAsString() << "\n";
    // Print blocks in the order as they appear in code for a stable ordering.
    for (const CFGBlock *B : *AC.getAnalysis<PostOrderCFGView>()) {
      llvm::dbgs() << "  Block B" << B->getBlockID() << ":\n";
      auto It = BlockToFactsMap.find(B);
      if (It != BlockToFactsMap.end()) {
        for (const Fact *F : It->second) {
          llvm::dbgs() << "    ";
          F->dump(llvm::dbgs(), LoanMgr, OriginMgr);
        }
      }
      llvm::dbgs() << "  End of Block\n";
    }
  }

  LoanManager &getLoanMgr() { return LoanMgr; }
  OriginManager &getOriginMgr() { return OriginMgr; }

private:
  LoanManager LoanMgr;
  OriginManager OriginMgr;
  llvm::DenseMap<const clang::CFGBlock *, llvm::SmallVector<const Fact *>>
      BlockToFactsMap;
  llvm::BumpPtrAllocator FactAllocator;
};

class FactGenerator : public ConstStmtVisitor<FactGenerator> {
  using Base = ConstStmtVisitor<FactGenerator>;

public:
  FactGenerator(FactManager &FactMgr, AnalysisDeclContext &AC)
      : FactMgr(FactMgr), AC(AC) {}

  void run() {
    llvm::TimeTraceScope TimeProfile("FactGenerator");
    // Iterate through the CFG blocks in reverse post-order to ensure that
    // initializations and destructions are processed in the correct sequence.
    for (const CFGBlock *Block : *AC.getAnalysis<PostOrderCFGView>()) {
      CurrentBlockFacts.clear();
      for (unsigned I = 0; I < Block->size(); ++I) {
        const CFGElement &Element = Block->Elements[I];
        if (std::optional<CFGStmt> CS = Element.getAs<CFGStmt>())
          Visit(CS->getStmt());
        else if (std::optional<CFGAutomaticObjDtor> DtorOpt =
                     Element.getAs<CFGAutomaticObjDtor>())
          handleDestructor(*DtorOpt);
      }
      FactMgr.addBlockFacts(Block, CurrentBlockFacts);
    }
  }

  void VisitDeclStmt(const DeclStmt *DS) {
    for (const Decl *D : DS->decls())
      if (const auto *VD = dyn_cast<VarDecl>(D))
        if (hasOrigin(VD))
          if (const Expr *InitExpr = VD->getInit())
            killAndFlowOrigin(*VD, *InitExpr);
  }

  void VisitDeclRefExpr(const DeclRefExpr *DRE) {
    handleUse(DRE);
    // For non-pointer/non-view types, a reference to the variable's storage
    // is a borrow. We create a loan for it.
    // For pointer/view types, we stick to the existing model for now and do
    // not create an extra origin for the l-value expression itself.

    // TODO: A single origin for a `DeclRefExpr` for a pointer or view type is
    // not sufficient to model the different levels of indirection. The current
    // single-origin model cannot distinguish between a loan to the variable's
    // storage and a loan to what it points to. A multi-origin model would be
    // required for this.
    if (!isPointerType(DRE->getType())) {
      if (const Loan *L = createLoan(DRE)) {
        OriginID ExprOID = FactMgr.getOriginMgr().getOrCreate(*DRE);
        CurrentBlockFacts.push_back(
            FactMgr.createFact<IssueFact>(L->ID, ExprOID));
      }
    }
  }

  void VisitCXXConstructExpr(const CXXConstructExpr *CCE) {
    if (isGslPointerType(CCE->getType())) {
      handleGSLPointerConstruction(CCE);
      return;
    }
  }

  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE) {
    // Specifically for conversion operators,
    // like `std::string_view p = std::string{};`
    if (isGslPointerType(MCE->getType()) &&
        isa<CXXConversionDecl>(MCE->getCalleeDecl())) {
      // The argument is the implicit object itself.
      handleFunctionCall(MCE, MCE->getMethodDecl(),
                         {MCE->getImplicitObjectArgument()},
                         /*IsGslConstruction=*/true);
    }
    if (const CXXMethodDecl *Method = MCE->getMethodDecl()) {
      // Construct the argument list, with the implicit 'this' object as the
      // first argument.
      llvm::SmallVector<const Expr *, 4> Args;
      Args.push_back(MCE->getImplicitObjectArgument());
      Args.append(MCE->getArgs(), MCE->getArgs() + MCE->getNumArgs());

      handleFunctionCall(MCE, Method, Args, /*IsGslConstruction=*/false);
    }
  }

  void VisitCallExpr(const CallExpr *CE) {
    handleFunctionCall(CE, CE->getDirectCallee(),
                       {CE->getArgs(), CE->getNumArgs()});
  }

  void VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *N) {
    /// TODO: Handle nullptr expr as a special 'null' loan. Uninitialized
    /// pointers can use the same type of loan.
    FactMgr.getOriginMgr().getOrCreate(*N);
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
    if (!hasOrigin(ICE))
      return;
    // An ImplicitCastExpr node itself gets an origin, which flows from the
    // origin of its sub-expression (after stripping its own parens/casts).
    killAndFlowOrigin(*ICE, *ICE->getSubExpr());
  }

  void VisitUnaryOperator(const UnaryOperator *UO) {
    if (UO->getOpcode() == UO_AddrOf) {
      const Expr *SubExpr = UO->getSubExpr();
      // Taking address of a pointer-type expression is not yet supported and
      // will be supported in multi-origin model.
      if (isPointerType(SubExpr->getType()))
        return;
      // The origin of an address-of expression (e.g., &x) is the origin of
      // its sub-expression (x). This fact will cause the dataflow analysis
      // to propagate any loans held by the sub-expression's origin to the
      // origin of this UnaryOperator expression.
      killAndFlowOrigin(*UO, *SubExpr);
    }
  }

  void VisitReturnStmt(const ReturnStmt *RS) {
    if (const Expr *RetExpr = RS->getRetValue()) {
      if (hasOrigin(RetExpr)) {
        OriginID OID = FactMgr.getOriginMgr().getOrCreate(*RetExpr);
        CurrentBlockFacts.push_back(
            FactMgr.createFact<ReturnOfOriginFact>(OID));
      }
    }
  }

  void VisitBinaryOperator(const BinaryOperator *BO) {
    if (BO->isAssignmentOp())
      handleAssignment(BO->getLHS(), BO->getRHS());
  }

  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *OCE) {
    // Assignment operators have special "kill-then-propagate" semantics
    // and are handled separately.
    if (OCE->isAssignmentOp() && OCE->getNumArgs() == 2) {
      handleAssignment(OCE->getArg(0), OCE->getArg(1));
      return;
    }
    handleFunctionCall(OCE, OCE->getDirectCallee(),
                       {OCE->getArgs(), OCE->getNumArgs()},
                       /*IsGslConstruction=*/false);
  }

  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *FCE) {
    // Check if this is a test point marker. If so, we are done with this
    // expression.
    if (handleTestPoint(FCE))
      return;
    if (isGslPointerType(FCE->getType()))
      killAndFlowOrigin(*FCE, *FCE->getSubExpr());
  }

  void VisitInitListExpr(const InitListExpr *ILE) {
    if (!hasOrigin(ILE))
      return;
    // For list initialization with a single element, like `View{...}`, the
    // origin of the list itself is the origin of its single element.
    if (ILE->getNumInits() == 1)
      killAndFlowOrigin(*ILE, *ILE->getInit(0));
  }

  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *MTE) {
    if (!hasOrigin(MTE))
      return;
    // A temporary object's origin is the same as the origin of the
    // expression that initializes it.
    killAndFlowOrigin(*MTE, *MTE->getSubExpr());
  }

  void handleDestructor(const CFGAutomaticObjDtor &DtorOpt) {
    /// TODO: Also handle trivial destructors (e.g., for `int`
    /// variables) which will never have a CFGAutomaticObjDtor node.
    /// TODO: Handle loans to temporaries.
    /// TODO: Consider using clang::CFG::BuildOptions::AddLifetime to reuse the
    /// lifetime ends.
    const VarDecl *DestructedVD = DtorOpt.getVarDecl();
    if (!DestructedVD)
      return;
    // Iterate through all loans to see if any expire.
    /// TODO(opt): Do better than a linear search to find loans associated with
    /// 'DestructedVD'.
    for (const Loan &L : FactMgr.getLoanMgr().getLoans()) {
      const AccessPath &LoanPath = L.Path;
      // Check if the loan is for a stack variable and if that variable
      // is the one being destructed.
      if (LoanPath.D == DestructedVD)
        CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(
            L.ID, DtorOpt.getTriggerStmt()->getEndLoc()));
    }
  }

private:
  static bool isGslPointerType(QualType QT) {
    if (const auto *RD = QT->getAsCXXRecordDecl()) {
      // We need to check the template definition for specializations.
      if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
        return CTSD->getSpecializedTemplate()
            ->getTemplatedDecl()
            ->hasAttr<PointerAttr>();
      return RD->hasAttr<PointerAttr>();
    }
    return false;
  }

  static bool isPointerType(QualType QT) {
    return QT->isPointerOrReferenceType() || isGslPointerType(QT);
  }
  // Check if a type has an origin.
  static bool hasOrigin(const Expr *E) {
    return E->isGLValue() || isPointerType(E->getType());
  }

  static bool hasOrigin(const VarDecl *VD) {
    return isPointerType(VD->getType());
  }

  void handleGSLPointerConstruction(const CXXConstructExpr *CCE) {
    assert(isGslPointerType(CCE->getType()));
    if (CCE->getNumArgs() != 1)
      return;
    if (hasOrigin(CCE->getArg(0)))
      killAndFlowOrigin(*CCE, *CCE->getArg(0));
    else
      // This could be a new borrow.
      handleFunctionCall(CCE, CCE->getConstructor(),
                         {CCE->getArgs(), CCE->getNumArgs()},
                         /*IsGslConstruction=*/true);
  }

  /// Checks if a call-like expression creates a borrow by passing a value to a
  /// reference parameter, creating an IssueFact if it does.
  /// \param IsGslConstruction True if this is a GSL construction where all
  ///   argument origins should flow to the returned origin.
  void handleFunctionCall(const Expr *Call, const FunctionDecl *FD,
                          ArrayRef<const Expr *> Args,
                          bool IsGslConstruction = false) {
    // Ignore functions returning values with no origin.
    if (!FD || !hasOrigin(Call))
      return;
    auto IsArgLifetimeBound = [FD](unsigned I) -> bool {
      const ParmVarDecl *PVD = nullptr;
      if (const auto *Method = dyn_cast<CXXMethodDecl>(FD);
          Method && Method->isInstance()) {
        if (I == 0)
          // For the 'this' argument, the attribute is on the method itself.
          return implicitObjectParamIsLifetimeBound(Method);
        if ((I - 1) < Method->getNumParams())
          // For explicit arguments, find the corresponding parameter
          // declaration.
          PVD = Method->getParamDecl(I - 1);
      } else if (I < FD->getNumParams())
        // For free functions or static methods.
        PVD = FD->getParamDecl(I);
      return PVD ? PVD->hasAttr<clang::LifetimeBoundAttr>() : false;
    };
    if (Args.empty())
      return;
    bool killedSrc = false;
    for (unsigned I = 0; I < Args.size(); ++I)
      if (IsGslConstruction || IsArgLifetimeBound(I)) {
        if (!killedSrc) {
          killedSrc = true;
          killAndFlowOrigin(*Call, *Args[I]);
        } else
          flowOrigin(*Call, *Args[I]);
      }
  }

  /// Creates a loan for the storage path of a given declaration reference.
  /// This function should be called whenever a DeclRefExpr represents a borrow.
  /// \param DRE The declaration reference expression that initiates the borrow.
  /// \return The new Loan on success, nullptr otherwise.
  const Loan *createLoan(const DeclRefExpr *DRE) {
    if (const auto *VD = dyn_cast<ValueDecl>(DRE->getDecl())) {
      AccessPath Path(VD);
      // The loan is created at the location of the DeclRefExpr.
      return &FactMgr.getLoanMgr().addLoan(Path, DRE);
    }
    return nullptr;
  }

  template <typename Destination, typename Source>
  void flowOrigin(const Destination &D, const Source &S) {
    OriginID DestOID = FactMgr.getOriginMgr().getOrCreate(D);
    OriginID SrcOID = FactMgr.getOriginMgr().get(S);
    CurrentBlockFacts.push_back(FactMgr.createFact<OriginFlowFact>(
        DestOID, SrcOID, /*KillDest=*/false));
  }

  template <typename Destination, typename Source>
  void killAndFlowOrigin(const Destination &D, const Source &S) {
    OriginID DestOID = FactMgr.getOriginMgr().getOrCreate(D);
    OriginID SrcOID = FactMgr.getOriginMgr().get(S);
    CurrentBlockFacts.push_back(
        FactMgr.createFact<OriginFlowFact>(DestOID, SrcOID, /*KillDest=*/true));
  }

  /// Checks if the expression is a `void("__lifetime_test_point_...")` cast.
  /// If so, creates a `TestPointFact` and returns true.
  bool handleTestPoint(const CXXFunctionalCastExpr *FCE) {
    if (!FCE->getType()->isVoidType())
      return false;

    const auto *SubExpr = FCE->getSubExpr()->IgnoreParenImpCasts();
    if (const auto *SL = dyn_cast<StringLiteral>(SubExpr)) {
      llvm::StringRef LiteralValue = SL->getString();
      const std::string Prefix = "__lifetime_test_point_";

      if (LiteralValue.starts_with(Prefix)) {
        StringRef Annotation = LiteralValue.drop_front(Prefix.length());
        CurrentBlockFacts.push_back(
            FactMgr.createFact<TestPointFact>(Annotation));
        return true;
      }
    }
    return false;
  }

  void handleAssignment(const Expr *LHSExpr, const Expr *RHSExpr) {
    if (!hasOrigin(LHSExpr))
      return;
    // Find the underlying variable declaration for the left-hand side.
    if (const auto *DRE_LHS =
            dyn_cast<DeclRefExpr>(LHSExpr->IgnoreParenImpCasts())) {
      markUseAsWrite(DRE_LHS);
      if (const auto *VD_LHS = dyn_cast<ValueDecl>(DRE_LHS->getDecl())) {
        // Kill the old loans of the destination origin and flow the new loans
        // from the source origin.
        killAndFlowOrigin(*VD_LHS, *RHSExpr);
      }
    }
  }

  // A DeclRefExpr will be treated as a use of the referenced decl. It will be
  // checked for use-after-free unless it is later marked as being written to
  // (e.g. on the left-hand side of an assignment).
  void handleUse(const DeclRefExpr *DRE) {
    if (isPointerType(DRE->getType())) {
      UseFact *UF = FactMgr.createFact<UseFact>(DRE);
      CurrentBlockFacts.push_back(UF);
      assert(!UseFacts.contains(DRE));
      UseFacts[DRE] = UF;
    }
  }

  void markUseAsWrite(const DeclRefExpr *DRE) {
    if (!isPointerType(DRE->getType()))
      return;
    assert(UseFacts.contains(DRE));
    UseFacts[DRE]->markAsWritten();
  }

  FactManager &FactMgr;
  AnalysisDeclContext &AC;
  llvm::SmallVector<Fact *> CurrentBlockFacts;
  // To distinguish between reads and writes for use-after-free checks, this map
  // stores the `UseFact` for each `DeclRefExpr`. We initially identify all
  // `DeclRefExpr`s as "read" uses. When an assignment is processed, the use
  // corresponding to the left-hand side is updated to be a "write", thereby
  // exempting it from the check.
  llvm::DenseMap<const DeclRefExpr *, UseFact *> UseFacts;
};

// ========================================================================= //
//                         Generic Dataflow Analysis
// ========================================================================= //

enum class Direction { Forward, Backward };

/// A `ProgramPoint` identifies a location in the CFG by pointing to a specific
/// `Fact`. identified by a lifetime-related event (`Fact`).
///
/// A `ProgramPoint` has "after" semantics: it represents the location
/// immediately after its corresponding `Fact`.
using ProgramPoint = const Fact *;

/// A generic, policy-based driver for dataflow analyses. It combines
/// the dataflow runner and the transferer logic into a single class hierarchy.
///
/// The derived class is expected to provide:
/// - A `Lattice` type.
/// - `StringRef getAnalysisName() const`
/// - `Lattice getInitialState();` The initial state of the analysis.
/// - `Lattice join(Lattice, Lattice);` Merges states from multiple CFG paths.
/// - `Lattice transfer(Lattice, const FactType&);` Defines how a single
///   lifetime-relevant `Fact` transforms the lattice state. Only overloads
///   for facts relevant to the analysis need to be implemented.
///
/// \tparam Derived The CRTP derived class that implements the specific
/// analysis.
/// \tparam LatticeType The dataflow lattice used by the analysis.
/// \tparam Dir The direction of the analysis (Forward or Backward).
/// TODO: Maybe use the dataflow framework! The framework might need changes
/// to support the current comparison done at block-entry.
template <typename Derived, typename LatticeType, Direction Dir>
class DataflowAnalysis {
public:
  using Lattice = LatticeType;
  using Base = DataflowAnalysis<Derived, Lattice, Dir>;

private:
  const CFG &Cfg;
  AnalysisDeclContext &AC;

  /// The dataflow state before a basic block is processed.
  llvm::DenseMap<const CFGBlock *, Lattice> InStates;
  /// The dataflow state after a basic block is processed.
  llvm::DenseMap<const CFGBlock *, Lattice> OutStates;
  /// The dataflow state at a Program Point.
  /// In a forward analysis, this is the state after the Fact at that point has
  /// been applied, while in a backward analysis, it is the state before.
  llvm::DenseMap<ProgramPoint, Lattice> PerPointStates;

  static constexpr bool isForward() { return Dir == Direction::Forward; }

protected:
  FactManager &AllFacts;

  explicit DataflowAnalysis(const CFG &C, AnalysisDeclContext &AC,
                            FactManager &F)
      : Cfg(C), AC(AC), AllFacts(F) {}

public:
  void run() {
    Derived &D = static_cast<Derived &>(*this);
    llvm::TimeTraceScope Time(D.getAnalysisName());

    using Worklist =
        std::conditional_t<Dir == Direction::Forward, ForwardDataflowWorklist,
                           BackwardDataflowWorklist>;
    Worklist W(Cfg, AC);

    const CFGBlock *Start = isForward() ? &Cfg.getEntry() : &Cfg.getExit();
    InStates[Start] = D.getInitialState();
    W.enqueueBlock(Start);

    while (const CFGBlock *B = W.dequeue()) {
      Lattice StateIn = *getInState(B);
      Lattice StateOut = transferBlock(B, StateIn);
      OutStates[B] = StateOut;
      for (const CFGBlock *AdjacentB : isForward() ? B->succs() : B->preds()) {
        if (!AdjacentB)
          continue;
        std::optional<Lattice> OldInState = getInState(AdjacentB);
        Lattice NewInState =
            !OldInState ? StateOut : D.join(*OldInState, StateOut);
        // Enqueue the adjacent block if its in-state has changed or if we have
        // never seen it.
        if (!OldInState || NewInState != *OldInState) {
          InStates[AdjacentB] = NewInState;
          W.enqueueBlock(AdjacentB);
        }
      }
    }
  }

protected:
  Lattice getState(ProgramPoint P) const { return PerPointStates.lookup(P); }

  std::optional<Lattice> getInState(const CFGBlock *B) const {
    auto It = InStates.find(B);
    if (It == InStates.end())
      return std::nullopt;
    return It->second;
  }

  Lattice getOutState(const CFGBlock *B) const { return OutStates.lookup(B); }

  void dump() const {
    const Derived *D = static_cast<const Derived *>(this);
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << D->getAnalysisName() << " results:\n";
    llvm::dbgs() << "==========================================\n";
    const CFGBlock &B = isForward() ? Cfg.getExit() : Cfg.getEntry();
    getOutState(&B).dump(llvm::dbgs());
  }

private:
  /// Computes the state at one end of a block by applying all its facts
  /// sequentially to a given state from the other end.
  Lattice transferBlock(const CFGBlock *Block, Lattice State) {
    auto Facts = AllFacts.getFacts(Block);
    if constexpr (isForward()) {
      for (const Fact *F : Facts) {
        State = transferFact(State, F);
        PerPointStates[F] = State;
      }
    } else {
      for (const Fact *F : llvm::reverse(Facts)) {
        // In backward analysis, capture the state before applying the fact.
        PerPointStates[F] = State;
        State = transferFact(State, F);
      }
    }
    return State;
  }

  Lattice transferFact(Lattice In, const Fact *F) {
    assert(F);
    Derived *D = static_cast<Derived *>(this);
    switch (F->getKind()) {
    case Fact::Kind::Issue:
      return D->transfer(In, *F->getAs<IssueFact>());
    case Fact::Kind::Expire:
      return D->transfer(In, *F->getAs<ExpireFact>());
    case Fact::Kind::OriginFlow:
      return D->transfer(In, *F->getAs<OriginFlowFact>());
    case Fact::Kind::ReturnOfOrigin:
      return D->transfer(In, *F->getAs<ReturnOfOriginFact>());
    case Fact::Kind::Use:
      return D->transfer(In, *F->getAs<UseFact>());
    case Fact::Kind::TestPoint:
      return D->transfer(In, *F->getAs<TestPointFact>());
    }
    llvm_unreachable("Unknown fact kind");
  }

public:
  Lattice transfer(Lattice In, const IssueFact &) { return In; }
  Lattice transfer(Lattice In, const ExpireFact &) { return In; }
  Lattice transfer(Lattice In, const OriginFlowFact &) { return In; }
  Lattice transfer(Lattice In, const ReturnOfOriginFact &) { return In; }
  Lattice transfer(Lattice In, const UseFact &) { return In; }
  Lattice transfer(Lattice In, const TestPointFact &) { return In; }
};

namespace utils {

/// Computes the union of two ImmutableSets.
template <typename T>
static llvm::ImmutableSet<T> join(llvm::ImmutableSet<T> A,
                                  llvm::ImmutableSet<T> B,
                                  typename llvm::ImmutableSet<T>::Factory &F) {
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  for (const T &E : B)
    A = F.add(A, E);
  return A;
}

/// Describes the strategy for joining two `ImmutableMap` instances, primarily
/// differing in how they handle keys that are unique to one of the maps.
///
/// A `Symmetric` join is universally correct, while an `Asymmetric` join
/// serves as a performance optimization. The latter is applicable only when the
/// join operation possesses a left identity element, allowing for a more
/// efficient, one-sided merge.
enum class JoinKind {
  /// A symmetric join applies the `JoinValues` operation to keys unique to
  /// either map, ensuring that values from both maps contribute to the result.
  Symmetric,
  /// An asymmetric join preserves keys unique to the first map as-is, while
  /// applying the `JoinValues` operation only to keys unique to the second map.
  Asymmetric,
};

/// Computes the key-wise union of two ImmutableMaps.
// TODO(opt): This key-wise join is a performance bottleneck. A more
// efficient merge could be implemented using a Patricia Trie or HAMT
// instead of the current AVL-tree-based ImmutableMap.
template <typename K, typename V, typename Joiner>
static llvm::ImmutableMap<K, V>
join(const llvm::ImmutableMap<K, V> &A, const llvm::ImmutableMap<K, V> &B,
     typename llvm::ImmutableMap<K, V>::Factory &F, Joiner JoinValues,
     JoinKind Kind) {
  if (A.getHeight() < B.getHeight())
    return join(B, A, F, JoinValues, Kind);

  // For each element in B, join it with the corresponding element in A
  // (or with an empty value if it doesn't exist in A).
  llvm::ImmutableMap<K, V> Res = A;
  for (const auto &Entry : B) {
    const K &Key = Entry.first;
    const V &ValB = Entry.second;
    Res = F.add(Res, Key, JoinValues(A.lookup(Key), &ValB));
  }
  if (Kind == JoinKind::Symmetric) {
    for (const auto &Entry : A) {
      const K &Key = Entry.first;
      const V &ValA = Entry.second;
      if (!B.contains(Key))
        Res = F.add(Res, Key, JoinValues(&ValA, nullptr));
    }
  }
  return Res;
}
} // namespace utils

// ========================================================================= //
//                          Loan Propagation Analysis
// ========================================================================= //

/// Represents the dataflow lattice for loan propagation.
///
/// This lattice tracks which loans each origin may hold at a given program
/// point.The lattice has a finite height: An origin's loan set is bounded by
/// the total number of loans in the function.
/// TODO(opt): To reduce the lattice size, propagate origins of declarations,
/// not expressions, because expressions are not visible across blocks.
struct LoanPropagationLattice {
  /// The map from an origin to the set of loans it contains.
  OriginLoanMap Origins = OriginLoanMap(nullptr);

  explicit LoanPropagationLattice(const OriginLoanMap &S) : Origins(S) {}
  LoanPropagationLattice() = default;

  bool operator==(const LoanPropagationLattice &Other) const {
    return Origins == Other.Origins;
  }
  bool operator!=(const LoanPropagationLattice &Other) const {
    return !(*this == Other);
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "LoanPropagationLattice State:\n";
    if (Origins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : Origins) {
      if (Entry.second.isEmpty())
        OS << "  Origin " << Entry.first << " contains no loans\n";
      for (const LoanID &LID : Entry.second)
        OS << "  Origin " << Entry.first << " contains Loan " << LID << "\n";
    }
  }
};

/// The analysis that tracks which loans belong to which origins.
class LoanPropagationAnalysis
    : public DataflowAnalysis<LoanPropagationAnalysis, LoanPropagationLattice,
                              Direction::Forward> {
  OriginLoanMap::Factory &OriginLoanMapFactory;
  LoanSet::Factory &LoanSetFactory;

public:
  LoanPropagationAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                          OriginLoanMap::Factory &OriginLoanMapFactory,
                          LoanSet::Factory &LoanSetFactory)
      : DataflowAnalysis(C, AC, F), OriginLoanMapFactory(OriginLoanMapFactory),
        LoanSetFactory(LoanSetFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges two lattices by taking the union of loans for each origin.
  // TODO(opt): Keep the state small by removing origins which become dead.
  Lattice join(Lattice A, Lattice B) {
    OriginLoanMap JoinedOrigins = utils::join(
        A.Origins, B.Origins, OriginLoanMapFactory,
        [&](const LoanSet *S1, const LoanSet *S2) {
          assert((S1 || S2) && "unexpectedly merging 2 empty sets");
          if (!S1)
            return *S2;
          if (!S2)
            return *S1;
          return utils::join(*S1, *S2, LoanSetFactory);
        },
        // Asymmetric join is a performance win. For origins present only on one
        // branch, the loan set can be carried over as-is.
        utils::JoinKind::Asymmetric);
    return Lattice(JoinedOrigins);
  }

  /// A new loan is issued to the origin. Old loans are erased.
  Lattice transfer(Lattice In, const IssueFact &F) {
    OriginID OID = F.getOriginID();
    LoanID LID = F.getLoanID();
    return LoanPropagationLattice(OriginLoanMapFactory.add(
        In.Origins, OID,
        LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID)));
  }

  /// A flow from source to destination. If `KillDest` is true, this replaces
  /// the destination's loans with the source's. Otherwise, the source's loans
  /// are merged into the destination's.
  Lattice transfer(Lattice In, const OriginFlowFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();

    LoanSet DestLoans =
        F.getKillDest() ? LoanSetFactory.getEmptySet() : getLoans(In, DestOID);
    LoanSet SrcLoans = getLoans(In, SrcOID);
    LoanSet MergedLoans = utils::join(DestLoans, SrcLoans, LoanSetFactory);

    return LoanPropagationLattice(
        OriginLoanMapFactory.add(In.Origins, DestOID, MergedLoans));
  }

  LoanSet getLoans(OriginID OID, ProgramPoint P) const {
    return getLoans(getState(P), OID);
  }

private:
  LoanSet getLoans(Lattice L, OriginID OID) const {
    if (auto *Loans = L.Origins.lookup(OID))
      return *Loans;
    return LoanSetFactory.getEmptySet();
  }
};

// ========================================================================= //
//                         Live Origins Analysis
// ========================================================================= //
//
// A backward dataflow analysis that determines which origins are "live" at each
// program point. An origin is "live" at a program point if there's a potential
// future use of the pointer it represents. Liveness is "generated" by a read of
// origin's loan set (e.g., a `UseFact`) and is "killed" (i.e., it stops being
// live) when its loan set is overwritten (e.g. a OriginFlow killing the
// destination origin).
//
// This information is used for detecting use-after-free errors, as it allows us
// to check if a live origin holds a loan to an object that has already expired.
// ========================================================================= //

/// Information about why an origin is live at a program point.
struct LivenessInfo {
  /// The use that makes the origin live. If liveness is propagated from
  /// multiple uses along different paths, this will point to the use appearing
  /// earlier in the translation unit.
  /// This is 'null' when the origin is not live.
  const UseFact *CausingUseFact;
  /// The kind of liveness of the origin.
  /// `Must`: The origin is live on all control-flow paths from the current
  /// point to the function's exit (i.e. the current point is dominated by a set
  /// of uses).
  /// `Maybe`: indicates it is live on some but not all paths.
  ///
  /// This determines the diagnostic's confidence level.
  /// `Must`-be-alive at expiration implies a definite use-after-free,
  /// while `Maybe`-be-alive suggests a potential one on some paths.
  LivenessKind Kind;

  LivenessInfo() : CausingUseFact(nullptr), Kind(LivenessKind::Dead) {}
  LivenessInfo(const UseFact *UF, LivenessKind K)
      : CausingUseFact(UF), Kind(K) {}

  bool operator==(const LivenessInfo &Other) const {
    return CausingUseFact == Other.CausingUseFact && Kind == Other.Kind;
  }
  bool operator!=(const LivenessInfo &Other) const { return !(*this == Other); }

  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddPointer(CausingUseFact);
    IDBuilder.Add(Kind);
  }
};

using LivenessMap = llvm::ImmutableMap<OriginID, LivenessInfo>;

/// The dataflow lattice for origin liveness analysis.
/// It tracks which origins are live, why they're live (which UseFact),
/// and the confidence level of that liveness.
struct LivenessLattice {
  LivenessMap LiveOrigins;

  LivenessLattice() : LiveOrigins(nullptr) {};

  explicit LivenessLattice(LivenessMap L) : LiveOrigins(L) {}

  bool operator==(const LivenessLattice &Other) const {
    return LiveOrigins == Other.LiveOrigins;
  }

  bool operator!=(const LivenessLattice &Other) const {
    return !(*this == Other);
  }

  void dump(llvm::raw_ostream &OS, const OriginManager &OM) const {
    if (LiveOrigins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : LiveOrigins) {
      OriginID OID = Entry.first;
      const LivenessInfo &Info = Entry.second;
      OS << "  ";
      OM.dump(OID, OS);
      OS << " is ";
      switch (Info.Kind) {
      case LivenessKind::Must:
        OS << "definitely";
        break;
      case LivenessKind::Maybe:
        OS << "maybe";
        break;
      case LivenessKind::Dead:
        llvm_unreachable("liveness kind of live origins should not be dead.");
      }
      OS << " live at this point\n";
    }
  }
};

/// The analysis that tracks which origins are live, with granular information
/// about the causing use fact and confidence level. This is a backward
/// analysis.
class LiveOriginAnalysis
    : public DataflowAnalysis<LiveOriginAnalysis, LivenessLattice,
                              Direction::Backward> {
  FactManager &FactMgr;
  LivenessMap::Factory &Factory;

public:
  LiveOriginAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                     LivenessMap::Factory &SF)
      : DataflowAnalysis(C, AC, F), FactMgr(F), Factory(SF) {}
  using DataflowAnalysis<LiveOriginAnalysis, Lattice,
                         Direction::Backward>::transfer;

  StringRef getAnalysisName() const { return "LiveOrigins"; }

  Lattice getInitialState() { return Lattice(Factory.getEmptyMap()); }

  /// Merges two lattices by combining liveness information.
  /// When the same origin has different confidence levels, we take the lower
  /// one.
  Lattice join(Lattice L1, Lattice L2) const {
    LivenessMap Merged = L1.LiveOrigins;
    // Take the earliest UseFact to make the join hermetic and commutative.
    auto CombineUseFact = [](const UseFact &A,
                             const UseFact &B) -> const UseFact * {
      return A.getUseExpr()->getExprLoc() < B.getUseExpr()->getExprLoc() ? &A
                                                                         : &B;
    };
    auto CombineLivenessKind = [](LivenessKind K1,
                                  LivenessKind K2) -> LivenessKind {
      assert(K1 != LivenessKind::Dead && "LivenessKind should not be dead.");
      assert(K2 != LivenessKind::Dead && "LivenessKind should not be dead.");
      // Only return "Must" if both paths are "Must", otherwise Maybe.
      if (K1 == LivenessKind::Must && K2 == LivenessKind::Must)
        return LivenessKind::Must;
      return LivenessKind::Maybe;
    };
    auto CombineLivenessInfo = [&](const LivenessInfo *L1,
                                   const LivenessInfo *L2) -> LivenessInfo {
      assert((L1 || L2) && "unexpectedly merging 2 empty sets");
      if (!L1)
        return LivenessInfo(L2->CausingUseFact, LivenessKind::Maybe);
      if (!L2)
        return LivenessInfo(L1->CausingUseFact, LivenessKind::Maybe);
      return LivenessInfo(
          CombineUseFact(*L1->CausingUseFact, *L2->CausingUseFact),
          CombineLivenessKind(L1->Kind, L2->Kind));
    };
    return Lattice(utils::join(
        L1.LiveOrigins, L2.LiveOrigins, Factory, CombineLivenessInfo,
        // A symmetric join is required here. If an origin is live on one
        // branch but not the other, its confidence must be demoted to `Maybe`.
        utils::JoinKind::Symmetric));
  }

  /// A read operation makes the origin live with definite confidence, as it
  /// dominates this program point. A write operation kills the liveness of
  /// the origin since it overwrites the value.
  Lattice transfer(Lattice In, const UseFact &UF) {
    OriginID OID = UF.getUsedOrigin(FactMgr.getOriginMgr());
    // Write kills liveness.
    if (UF.isWritten())
      return Lattice(Factory.remove(In.LiveOrigins, OID));
    // Read makes origin live with definite confidence (dominates this point).
    return Lattice(Factory.add(In.LiveOrigins, OID,
                               LivenessInfo(&UF, LivenessKind::Must)));
  }

  /// Issuing a new loan to an origin kills its liveness.
  Lattice transfer(Lattice In, const IssueFact &IF) {
    return Lattice(Factory.remove(In.LiveOrigins, IF.getOriginID()));
  }

  /// An OriginFlow kills the liveness of the destination origin if `KillDest`
  /// is true. Otherwise, it propagates liveness from destination to source.
  Lattice transfer(Lattice In, const OriginFlowFact &OF) {
    if (!OF.getKillDest())
      return In;
    return Lattice(Factory.remove(In.LiveOrigins, OF.getDestOriginID()));
  }

  LivenessMap getLiveOrigins(ProgramPoint P) const {
    return getState(P).LiveOrigins;
  }

  // Dump liveness values on all test points in the program.
  void dump(llvm::raw_ostream &OS, const LifetimeSafetyAnalysis &LSA) const {
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << getAnalysisName() << " results:\n";
    llvm::dbgs() << "==========================================\n";
    for (const auto &Entry : LSA.getTestPoints()) {
      OS << "TestPoint: " << Entry.getKey() << "\n";
      getState(Entry.getValue()).dump(OS, FactMgr.getOriginMgr());
    }
  }
};

// ========================================================================= //
//                       Lifetime checker and Error reporter
// ========================================================================= //

/// Struct to store the complete context for a potential lifetime violation.
struct PendingWarning {
  SourceLocation ExpiryLoc; // Where the loan expired.
  const Expr *UseExpr;      // Where the origin holding this loan was used.
  Confidence ConfidenceLevel;
};

class LifetimeChecker {
private:
  llvm::DenseMap<LoanID, PendingWarning> FinalWarningsMap;
  LoanPropagationAnalysis &LoanPropagation;
  LiveOriginAnalysis &LiveOrigins;
  FactManager &FactMgr;
  AnalysisDeclContext &ADC;
  LifetimeSafetyReporter *Reporter;

public:
  LifetimeChecker(LoanPropagationAnalysis &LPA, LiveOriginAnalysis &LOA,
                  FactManager &FM, AnalysisDeclContext &ADC,
                  LifetimeSafetyReporter *Reporter)
      : LoanPropagation(LPA), LiveOrigins(LOA), FactMgr(FM), ADC(ADC),
        Reporter(Reporter) {}

  void run() {
    llvm::TimeTraceScope TimeProfile("LifetimeChecker");
    for (const CFGBlock *B : *ADC.getAnalysis<PostOrderCFGView>())
      for (const Fact *F : FactMgr.getFacts(B))
        if (const auto *EF = F->getAs<ExpireFact>())
          checkExpiry(EF);
    issuePendingWarnings();
  }

  /// Checks for use-after-free errors when a loan expires.
  ///
  /// This method examines all live origins at the expiry point and determines
  /// if any of them hold the expiring loan. If so, it creates a pending
  /// warning with the appropriate confidence level based on the liveness
  /// information. The confidence reflects whether the origin is definitely
  /// or maybe live at this point.
  ///
  /// Note: This implementation considers only the confidence of origin
  /// liveness. Future enhancements could also consider the confidence of loan
  /// propagation (e.g., a loan may only be held on some execution paths).
  void checkExpiry(const ExpireFact *EF) {
    LoanID ExpiredLoan = EF->getLoanID();
    LivenessMap Origins = LiveOrigins.getLiveOrigins(EF);
    Confidence CurConfidence = Confidence::None;
    const UseFact *BadUse = nullptr;
    for (auto &[OID, LiveInfo] : Origins) {
      LoanSet HeldLoans = LoanPropagation.getLoans(OID, EF);
      if (!HeldLoans.contains(ExpiredLoan))
        continue;
      // Loan is defaulted.
      Confidence NewConfidence = livenessKindToConfidence(LiveInfo.Kind);
      if (CurConfidence < NewConfidence) {
        CurConfidence = NewConfidence;
        BadUse = LiveInfo.CausingUseFact;
      }
    }
    if (!BadUse)
      return;
    // We have a use-after-free.
    Confidence LastConf = FinalWarningsMap.lookup(ExpiredLoan).ConfidenceLevel;
    if (LastConf >= CurConfidence)
      return;
    FinalWarningsMap[ExpiredLoan] = {/*ExpiryLoc=*/EF->getExpiryLoc(),
                                     /*UseExpr=*/BadUse->getUseExpr(),
                                     /*ConfidenceLevel=*/CurConfidence};
  }

  static Confidence livenessKindToConfidence(LivenessKind K) {
    switch (K) {
    case LivenessKind::Must:
      return Confidence::Definite;
    case LivenessKind::Maybe:
      return Confidence::Maybe;
    case LivenessKind::Dead:
      return Confidence::None;
    }
    llvm_unreachable("unknown liveness kind");
  }

  void issuePendingWarnings() {
    if (!Reporter)
      return;
    for (const auto &[LID, Warning] : FinalWarningsMap) {
      const Loan &L = FactMgr.getLoanMgr().getLoan(LID);
      const Expr *IssueExpr = L.IssueExpr;
      Reporter->reportUseAfterFree(IssueExpr, Warning.UseExpr,
                                   Warning.ExpiryLoc, Warning.ConfidenceLevel);
    }
  }
};

// ========================================================================= //
//                  LifetimeSafetyAnalysis Class Implementation
// ========================================================================= //

/// An object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  llvm::BumpPtrAllocator Allocator;
  OriginLoanMap::Factory OriginMapFactory{Allocator, /*canonicalize=*/false};
  LoanSet::Factory LoanSetFactory{Allocator, /*canonicalize=*/false};
  LivenessMap::Factory LivenessMapFactory{Allocator, /*canonicalize=*/false};
};

// We need this here for unique_ptr with forward declared class.
LifetimeSafetyAnalysis::~LifetimeSafetyAnalysis() = default;

LifetimeSafetyAnalysis::LifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                                               LifetimeSafetyReporter *Reporter)
    : AC(AC), Reporter(Reporter), Factory(std::make_unique<LifetimeFactory>()),
      FactMgr(std::make_unique<FactManager>()) {}

void LifetimeSafetyAnalysis::run() {
  llvm::TimeTraceScope TimeProfile("LifetimeSafetyAnalysis");

  const CFG &Cfg = *AC.getCFG();
  DEBUG_WITH_TYPE("PrintCFG", Cfg.dump(AC.getASTContext().getLangOpts(),
                                       /*ShowColors=*/true));

  FactGenerator FactGen(*FactMgr, AC);
  FactGen.run();
  DEBUG_WITH_TYPE("LifetimeFacts", FactMgr->dump(Cfg, AC));

  /// TODO(opt): Consider optimizing individual blocks before running the
  /// dataflow analysis.
  /// 1. Expression Origins: These are assigned once and read at most once,
  ///    forming simple chains. These chains can be compressed into a single
  ///    assignment.
  /// 2. Block-Local Loans: Origins of expressions are never read by other
  ///    blocks; only Decls are visible.  Therefore, loans in a block that
  ///    never reach an Origin associated with a Decl can be safely dropped by
  ///    the analysis.
  /// 3. Collapse ExpireFacts belonging to same source location into a single
  ///    Fact.
  LoanPropagation = std::make_unique<LoanPropagationAnalysis>(
      Cfg, AC, *FactMgr, Factory->OriginMapFactory, Factory->LoanSetFactory);
  LoanPropagation->run();

  LiveOrigins = std::make_unique<LiveOriginAnalysis>(
      Cfg, AC, *FactMgr, Factory->LivenessMapFactory);
  LiveOrigins->run();
  DEBUG_WITH_TYPE("LiveOrigins", LiveOrigins->dump(llvm::dbgs(), *this));

  LifetimeChecker Checker(*LoanPropagation, *LiveOrigins, *FactMgr, AC,
                          Reporter);
  Checker.run();
}

LoanSet LifetimeSafetyAnalysis::getLoansAtPoint(OriginID OID,
                                                ProgramPoint PP) const {
  assert(LoanPropagation && "Analysis has not been run.");
  return LoanPropagation->getLoans(OID, PP);
}

std::optional<OriginID>
LifetimeSafetyAnalysis::getOriginIDForDecl(const ValueDecl *D) const {
  assert(FactMgr && "FactManager not initialized");
  // This assumes the OriginManager's `get` can find an existing origin.
  // We might need a `find` method on OriginManager to avoid `getOrCreate` logic
  // in a const-query context if that becomes an issue.
  return FactMgr->getOriginMgr().get(*D);
}

std::vector<LoanID>
LifetimeSafetyAnalysis::getLoanIDForVar(const VarDecl *VD) const {
  assert(FactMgr && "FactManager not initialized");
  std::vector<LoanID> Result;
  for (const Loan &L : FactMgr->getLoanMgr().getLoans())
    if (L.Path.D == VD)
      Result.push_back(L.ID);
  return Result;
}

std::vector<std::pair<OriginID, LivenessKind>>
LifetimeSafetyAnalysis::getLiveOriginsAtPoint(ProgramPoint PP) const {
  assert(LiveOrigins && "LiveOriginAnalysis has not been run.");
  std::vector<std::pair<OriginID, LivenessKind>> Result;
  for (auto &[OID, Info] : LiveOrigins->getLiveOrigins(PP))
    Result.push_back({OID, Info.Kind});
  return Result;
}

llvm::StringMap<ProgramPoint> LifetimeSafetyAnalysis::getTestPoints() const {
  assert(FactMgr && "FactManager not initialized");
  llvm::StringMap<ProgramPoint> AnnotationToPointMap;
  for (const CFGBlock *Block : *AC.getCFG()) {
    for (const Fact *F : FactMgr->getFacts(Block)) {
      if (const auto *TPF = F->getAs<TestPointFact>()) {
        StringRef PointName = TPF->getAnnotation();
        assert(AnnotationToPointMap.find(PointName) ==
                   AnnotationToPointMap.end() &&
               "more than one test points with the same name");
        AnnotationToPointMap[PointName] = F;
      }
    }
  }
  return AnnotationToPointMap;
}
} // namespace internal

void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetyReporter *Reporter) {
  internal::LifetimeSafetyAnalysis Analysis(AC, Reporter);
  Analysis.run();
}
} // namespace clang::lifetimes
