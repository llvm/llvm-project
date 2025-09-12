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
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TimeProfiler.h"
#include <cstdint>
#include <memory>

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
    AssignOrigin,
    /// An origin escapes the function by flowing into the return value.
    ReturnOfOrigin,
    /// An origin is used (eg. dereferencing a pointer).
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

class AssignOriginFact : public Fact {
  OriginID OIDDest;
  OriginID OIDSrc;

public:
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::AssignOrigin;
  }

  AssignOriginFact(OriginID OIDDest, OriginID OIDSrc)
      : Fact(Kind::AssignOrigin), OIDDest(OIDDest), OIDSrc(OIDSrc) {}
  OriginID getDestOriginID() const { return OIDDest; }
  OriginID getSrcOriginID() const { return OIDSrc; }
  void dump(llvm::raw_ostream &OS, const LoanManager &,
            const OriginManager &OM) const override {
    OS << "AssignOrigin (Dest: ";
    OM.dump(getDestOriginID(), OS);
    OS << ", Src: ";
    OM.dump(getSrcOriginID(), OS);
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
            addAssignOriginFact(*VD, *InitExpr);
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
    addAssignOriginFact(*ICE, *ICE->getSubExpr());
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
      addAssignOriginFact(*UO, *SubExpr);
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
    if (OCE->isAssignmentOp() && OCE->getNumArgs() == 2)
      handleAssignment(OCE->getArg(0), OCE->getArg(1));
  }

  void VisitCXXFunctionalCastExpr(const CXXFunctionalCastExpr *FCE) {
    // Check if this is a test point marker. If so, we are done with this
    // expression.
    if (VisitTestPoint(FCE))
      return;
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
  static bool isPointerType(QualType QT) {
    return QT->isPointerOrReferenceType();
  }

  // Check if a type has an origin.
  static bool hasOrigin(const Expr *E) {
    return E->isGLValue() || isPointerType(E->getType());
  }

  static bool hasOrigin(const VarDecl *VD) {
    return isPointerType(VD->getType());
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
  void addAssignOriginFact(const Destination &D, const Source &S) {
    OriginID DestOID = FactMgr.getOriginMgr().getOrCreate(D);
    OriginID SrcOID = FactMgr.getOriginMgr().get(S);
    CurrentBlockFacts.push_back(
        FactMgr.createFact<AssignOriginFact>(DestOID, SrcOID));
  }

  /// Checks if the expression is a `void("__lifetime_test_point_...")` cast.
  /// If so, creates a `TestPointFact` and returns true.
  bool VisitTestPoint(const CXXFunctionalCastExpr *FCE) {
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
      if (const auto *VD_LHS = dyn_cast<ValueDecl>(DRE_LHS->getDecl()))
        // We are interested in assignments like `ptr1 = ptr2` or `ptr = &var`.
        // LHS must be a pointer/reference type that can be an origin. RHS must
        // also represent an origin (either another pointer/ref or an
        // address-of).
        addAssignOriginFact(*VD_LHS, *RHSExpr);
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

    llvm::SmallBitVector Visited(Cfg.getNumBlockIDs() + 1);

    while (const CFGBlock *B = W.dequeue()) {
      Lattice StateIn = getInState(B);
      Lattice StateOut = transferBlock(B, StateIn);
      OutStates[B] = StateOut;
      Visited.set(B->getBlockID());
      for (const CFGBlock *AdjacentB : isForward() ? B->succs() : B->preds()) {
        if (!AdjacentB)
          continue;
        Lattice OldInState = getInState(AdjacentB);
        Lattice NewInState = D.join(OldInState, StateOut);
        // Enqueue the adjacent block if its in-state has changed or if we have
        // never visited it.
        if (!Visited.test(AdjacentB->getBlockID()) ||
            NewInState != OldInState) {
          InStates[AdjacentB] = NewInState;
          W.enqueueBlock(AdjacentB);
        }
      }
    }
  }

protected:
  Lattice getState(ProgramPoint P) const { return PerPointStates.lookup(P); }

  Lattice getInState(const CFGBlock *B) const { return InStates.lookup(B); }

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
    case Fact::Kind::AssignOrigin:
      return D->transfer(In, *F->getAs<AssignOriginFact>());
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
  Lattice transfer(Lattice In, const AssignOriginFact &) { return In; }
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

/// Checks if set A is a subset of set B.
template <typename T>
static bool isSubsetOf(const llvm::ImmutableSet<T> &A,
                       const llvm::ImmutableSet<T> &B) {
  // Empty set is a subset of all sets.
  if (A.isEmpty())
    return true;

  for (const T &Elem : A)
    if (!B.contains(Elem))
      return false;
  return true;
}

/// Computes the key-wise union of two ImmutableMaps.
// TODO(opt): This key-wise join is a performance bottleneck. A more
// efficient merge could be implemented using a Patricia Trie or HAMT
// instead of the current AVL-tree-based ImmutableMap.
template <typename K, typename V, typename Joiner>
static llvm::ImmutableMap<K, V>
join(llvm::ImmutableMap<K, V> A, llvm::ImmutableMap<K, V> B,
     typename llvm::ImmutableMap<K, V>::Factory &F, Joiner JoinValues) {
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);

  // For each element in B, join it with the corresponding element in A
  // (or with an empty value if it doesn't exist in A).
  for (const auto &Entry : B) {
    const K &Key = Entry.first;
    const V &ValB = Entry.second;
    if (const V *ValA = A.lookup(Key))
      A = F.add(A, Key, JoinValues(*ValA, ValB));
    else
      A = F.add(A, Key, ValB);
  }
  return A;
}
} // namespace utils

// ========================================================================= //
//                          Loan Propagation Analysis
// ========================================================================= //

using OriginLoanMap = llvm::ImmutableMap<OriginID, LoanSet>;
using ExpiredLoanMap = llvm::ImmutableMap<LoanID, const ExpireFact *>;

/// An object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  OriginLoanMap::Factory OriginMapFactory;
  LoanSet::Factory LoanSetFactory;
  ExpiredLoanMap::Factory ExpiredLoanMapFactory;
};

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
                          LifetimeFactory &LFactory)
      : DataflowAnalysis(C, AC, F),
        OriginLoanMapFactory(LFactory.OriginMapFactory),
        LoanSetFactory(LFactory.LoanSetFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges two lattices by taking the union of loans for each origin.
  // TODO(opt): Keep the state small by removing origins which become dead.
  Lattice join(Lattice A, Lattice B) {
    OriginLoanMap JoinedOrigins =
        utils::join(A.Origins, B.Origins, OriginLoanMapFactory,
                    [&](LoanSet S1, LoanSet S2) {
                      return utils::join(S1, S2, LoanSetFactory);
                    });
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

  /// The destination origin's loan set is replaced by the source's.
  /// This implicitly "resets" the old loans of the destination.
  Lattice transfer(Lattice In, const AssignOriginFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();
    LoanSet SrcLoans = getLoans(In, SrcOID);
    return LoanPropagationLattice(
        OriginLoanMapFactory.add(In.Origins, DestOID, SrcLoans));
  }

  LoanSet getLoans(OriginID OID, ProgramPoint P) {
    return getLoans(getState(P), OID);
  }

private:
  LoanSet getLoans(Lattice L, OriginID OID) {
    if (auto *Loans = L.Origins.lookup(OID))
      return *Loans;
    return LoanSetFactory.getEmptySet();
  }
};

// ========================================================================= //
//                         Expired Loans Analysis
// ========================================================================= //

/// The dataflow lattice for tracking the set of expired loans.
struct ExpiredLattice {
  /// Map from an expired `LoanID` to the `ExpireFact` that made it expire.
  ExpiredLoanMap Expired;

  ExpiredLattice() : Expired(nullptr) {};
  explicit ExpiredLattice(ExpiredLoanMap M) : Expired(M) {}

  bool operator==(const ExpiredLattice &Other) const {
    return Expired == Other.Expired;
  }
  bool operator!=(const ExpiredLattice &Other) const {
    return !(*this == Other);
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "ExpiredLattice State:\n";
    if (Expired.isEmpty())
      OS << "  <empty>\n";
    for (const auto &[ID, _] : Expired)
      OS << "  Loan " << ID << " is expired\n";
  }
};

/// The analysis that tracks which loans have expired.
class ExpiredLoansAnalysis
    : public DataflowAnalysis<ExpiredLoansAnalysis, ExpiredLattice,
                              Direction::Forward> {

  ExpiredLoanMap::Factory &Factory;

public:
  ExpiredLoansAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                       LifetimeFactory &Factory)
      : DataflowAnalysis(C, AC, F), Factory(Factory.ExpiredLoanMapFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "ExpiredLoans"; }

  Lattice getInitialState() { return Lattice(Factory.getEmptyMap()); }

  /// Merges two lattices by taking the union of the two expired loans.
  Lattice join(Lattice L1, Lattice L2) {
    return Lattice(
        utils::join(L1.Expired, L2.Expired, Factory,
                    // Take the last expiry fact to make this hermetic.
                    [](const ExpireFact *F1, const ExpireFact *F2) {
                      return F1->getExpiryLoc() > F2->getExpiryLoc() ? F1 : F2;
                    }));
  }

  Lattice transfer(Lattice In, const ExpireFact &F) {
    return Lattice(Factory.add(In.Expired, F.getLoanID(), &F));
  }

  // Removes the loan from the set of expired loans.
  //
  // When a loan is re-issued (e.g., in a loop), it is no longer considered
  // expired. A loan can be in the expired set at the point of issue due to
  // the dataflow state from a previous loop iteration being propagated along
  // a backedge in the CFG.
  //
  // Note: This has a subtle false-negative though where a loan from previous
  // iteration is not overwritten by a reissue. This needs careful tracking
  // of loans "across iterations" which can be considered for future
  // enhancements.
  //
  //    void foo(int safe) {
  //      int* p = &safe;
  //      int* q = &safe;
  //      while (condition()) {
  //        int x = 1;
  //        p = &x;    // A loan to 'x' is issued to 'p' in every iteration.
  //        if (condition()) {
  //          q = p;
  //        }
  //        (void)*p; // OK  — 'p' points to 'x' from new iteration.
  //        (void)*q; // UaF - 'q' still points to 'x' from previous iteration
  //                  // which is now destroyed.
  //      }
  // }
  Lattice transfer(Lattice In, const IssueFact &F) {
    return Lattice(Factory.remove(In.Expired, F.getLoanID()));
  }

  ExpiredLoanMap getExpiredLoans(ProgramPoint P) { return getState(P).Expired; }
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
  ExpiredLoansAnalysis &ExpiredLoans;
  FactManager &FactMgr;
  AnalysisDeclContext &ADC;
  LifetimeSafetyReporter *Reporter;

public:
  LifetimeChecker(LoanPropagationAnalysis &LPA, ExpiredLoansAnalysis &ELA,
                  FactManager &FM, AnalysisDeclContext &ADC,
                  LifetimeSafetyReporter *Reporter)
      : LoanPropagation(LPA), ExpiredLoans(ELA), FactMgr(FM), ADC(ADC),
        Reporter(Reporter) {}

  void run() {
    llvm::TimeTraceScope TimeProfile("LifetimeChecker");
    for (const CFGBlock *B : *ADC.getAnalysis<PostOrderCFGView>())
      for (const Fact *F : FactMgr.getFacts(B))
        if (const auto *UF = F->getAs<UseFact>())
          checkUse(UF);
    issuePendingWarnings();
  }

  /// Checks for use-after-free errors for a given use of an Origin.
  ///
  /// This method is called for each 'UseFact' identified in the control flow
  /// graph. It determines if the loans held by the used origin have expired
  /// at the point of use.
  void checkUse(const UseFact *UF) {
    if (UF->isWritten())
      return;
    OriginID O = UF->getUsedOrigin(FactMgr.getOriginMgr());

    // Get the set of loans that the origin might hold at this program point.
    LoanSet HeldLoans = LoanPropagation.getLoans(O, UF);

    // Get the set of all loans that have expired at this program point.
    ExpiredLoanMap AllExpiredLoans = ExpiredLoans.getExpiredLoans(UF);

    // If the pointer holds no loans or no loans have expired, there's nothing
    // to check.
    if (HeldLoans.isEmpty() || AllExpiredLoans.isEmpty())
      return;

    // Identify loans that which have expired but are held by the pointer. Using
    // them is a use-after-free.
    llvm::SmallVector<LoanID> DefaultedLoans;
    // A definite UaF error occurs if all loans the origin might hold have
    // expired.
    bool IsDefiniteError = true;
    for (LoanID L : HeldLoans) {
      if (AllExpiredLoans.contains(L))
        DefaultedLoans.push_back(L);
      else
        // If at least one loan is not expired, this use is not a definite UaF.
        IsDefiniteError = false;
    }
    // If there are no defaulted loans, the use is safe.
    if (DefaultedLoans.empty())
      return;

    // Determine the confidence level of the error (definite or maybe).
    Confidence CurrentConfidence =
        IsDefiniteError ? Confidence::Definite : Confidence::Maybe;

    // For each expired loan, create a pending warning.
    for (LoanID DefaultedLoan : DefaultedLoans) {
      // If we already have a warning for this loan with a higher or equal
      // confidence, skip this one.
      if (FinalWarningsMap.count(DefaultedLoan) &&
          CurrentConfidence <= FinalWarningsMap[DefaultedLoan].ConfidenceLevel)
        continue;

      auto *EF = AllExpiredLoans.lookup(DefaultedLoan);
      assert(EF && "Could not find ExpireFact for an expired loan.");

      FinalWarningsMap[DefaultedLoan] = {/*ExpiryLoc=*/(*EF)->getExpiryLoc(),
                                         /*UseExpr=*/UF->getUseExpr(),
                                         /*ConfidenceLevel=*/CurrentConfidence};
    }
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
  LoanPropagation =
      std::make_unique<LoanPropagationAnalysis>(Cfg, AC, *FactMgr, *Factory);
  LoanPropagation->run();

  ExpiredLoans =
      std::make_unique<ExpiredLoansAnalysis>(Cfg, AC, *FactMgr, *Factory);
  ExpiredLoans->run();

  LifetimeChecker Checker(*LoanPropagation, *ExpiredLoans, *FactMgr, AC,
                          Reporter);
  Checker.run();
}

LoanSet LifetimeSafetyAnalysis::getLoansAtPoint(OriginID OID,
                                                ProgramPoint PP) const {
  assert(LoanPropagation && "Analysis has not been run.");
  return LoanPropagation->getLoans(OID, PP);
}

std::vector<LoanID>
LifetimeSafetyAnalysis::getExpiredLoansAtPoint(ProgramPoint PP) const {
  assert(ExpiredLoans && "ExpiredLoansAnalysis has not been run.");
  std::vector<LoanID> Result;
  for (const auto &pair : ExpiredLoans->getExpiredLoans(PP))
    Result.push_back(pair.first);
  return Result;
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
