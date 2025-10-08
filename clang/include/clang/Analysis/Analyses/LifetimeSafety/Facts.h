#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTS_H

#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeSafety.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <cstdint>
#include <optional>

namespace clang::lifetimes {
namespace internal {
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
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_FACTS_H
