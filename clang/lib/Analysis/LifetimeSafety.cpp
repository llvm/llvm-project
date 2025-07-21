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

namespace clang {
namespace {

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable.
/// TODO: Model access paths of other types, e.g., s.field, heap and globals.
struct AccessPath {
  const clang::ValueDecl *D;

  AccessPath(const clang::ValueDecl *D) : D(D) {}
};

/// A generic, type-safe wrapper for an ID, distinguished by its `Tag` type.
/// Used for giving ID to loans and origins.
template <typename Tag> struct ID {
  uint32_t Value = 0;

  bool operator==(const ID<Tag> &Other) const { return Value == Other.Value; }
  bool operator!=(const ID<Tag> &Other) const { return !(*this == Other); }
  bool operator<(const ID<Tag> &Other) const { return Value < Other.Value; }
  ID<Tag> operator++(int) {
    ID<Tag> Tmp = *this;
    ++Value;
    return Tmp;
  }
  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddInteger(Value);
  }
};

template <typename Tag>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, ID<Tag> ID) {
  return OS << ID.Value;
}

using LoanID = ID<struct LoanTag>;
using OriginID = ID<struct OriginTag>;

/// Information about a single borrow, or "Loan". A loan is created when a
/// reference or pointer is created.
struct Loan {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
  LoanID ID;
  AccessPath Path;
  SourceLocation IssueLoc;

  Loan(LoanID id, AccessPath path, SourceLocation loc)
      : ID(id), Path(path), IssueLoc(loc) {}
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

  Loan &addLoan(AccessPath Path, SourceLocation Loc) {
    AllLoans.emplace_back(getNextLoanID(), Path, Loc);
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

  OriginID get(const Expr &E) {
    // Origin of DeclRefExpr is that of the declaration it refers to.
    if (const auto *DRE = dyn_cast<DeclRefExpr>(&E))
      return get(*DRE->getDecl());
    auto It = ExprToOriginID.find(&E);
    // TODO: This should be an assert(It != ExprToOriginID.end()). The current
    // implementation falls back to getOrCreate to avoid crashing on
    // yet-unhandled pointer expressions, creating an empty origin for them.
    if (It == ExprToOriginID.end())
      return getOrCreate(E);

    return It->second;
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

    if (const auto *DRE = dyn_cast<DeclRefExpr>(&E)) {
      // Origin of DeclRefExpr is that of the declaration it refers to.
      return getOrCreate(*DRE->getDecl());
    }
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
    ReturnOfOrigin
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

  virtual void dump(llvm::raw_ostream &OS) const {
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
  void dump(llvm::raw_ostream &OS) const override {
    OS << "Issue (LoanID: " << getLoanID() << ", OriginID: " << getOriginID()
       << ")\n";
  }
};

class ExpireFact : public Fact {
  LoanID LID;

public:
  static bool classof(const Fact *F) { return F->getKind() == Kind::Expire; }

  ExpireFact(LoanID LID) : Fact(Kind::Expire), LID(LID) {}
  LoanID getLoanID() const { return LID; }
  void dump(llvm::raw_ostream &OS) const override {
    OS << "Expire (LoanID: " << getLoanID() << ")\n";
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
  void dump(llvm::raw_ostream &OS) const override {
    OS << "AssignOrigin (DestID: " << getDestOriginID()
       << ", SrcID: " << getSrcOriginID() << ")\n";
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
  void dump(llvm::raw_ostream &OS) const override {
    OS << "ReturnOfOrigin (OriginID: " << getReturnedOriginID() << ")\n";
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
          F->dump(llvm::dbgs());
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
        if (hasOrigin(VD->getType()))
          if (const Expr *InitExpr = VD->getInit())
            addAssignOriginFact(*VD, *InitExpr);
  }

  void VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *N) {
    /// TODO: Handle nullptr expr as a special 'null' loan. Uninitialized
    /// pointers can use the same type of loan.
    FactMgr.getOriginMgr().getOrCreate(*N);
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
    if (!hasOrigin(ICE->getType()))
      return;
    Visit(ICE->getSubExpr());
    // An ImplicitCastExpr node itself gets an origin, which flows from the
    // origin of its sub-expression (after stripping its own parens/casts).
    // TODO: Consider if this is actually useful in practice. Alternatively, we
    // could directly use the sub-expression's OriginID instead of creating a
    // new one.
    addAssignOriginFact(*ICE, *ICE->getSubExpr());
  }

  void VisitUnaryOperator(const UnaryOperator *UO) {
    if (UO->getOpcode() == UO_AddrOf) {
      const Expr *SubExpr = UO->getSubExpr();
      if (const auto *DRE = dyn_cast<DeclRefExpr>(SubExpr)) {
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          // Check if it's a local variable.
          if (VD->hasLocalStorage()) {
            OriginID OID = FactMgr.getOriginMgr().getOrCreate(*UO);
            AccessPath AddrOfLocalVarPath(VD);
            const Loan &L = FactMgr.getLoanMgr().addLoan(AddrOfLocalVarPath,
                                                         UO->getOperatorLoc());
            CurrentBlockFacts.push_back(
                FactMgr.createFact<IssueFact>(L.ID, OID));
          }
        }
      }
    }
  }

  void VisitReturnStmt(const ReturnStmt *RS) {
    if (const Expr *RetExpr = RS->getRetValue()) {
      if (hasOrigin(RetExpr->getType())) {
        OriginID OID = FactMgr.getOriginMgr().getOrCreate(*RetExpr);
        CurrentBlockFacts.push_back(
            FactMgr.createFact<ReturnOfOriginFact>(OID));
      }
    }
  }

  void VisitBinaryOperator(const BinaryOperator *BO) {
    if (BO->isAssignmentOp()) {
      const Expr *LHSExpr = BO->getLHS();
      const Expr *RHSExpr = BO->getRHS();

      // We are interested in assignments like `ptr1 = ptr2` or `ptr = &var`
      // LHS must be a pointer/reference type that can be an origin.
      // RHS must also represent an origin (either another pointer/ref or an
      // address-of).
      if (const auto *DRE_LHS = dyn_cast<DeclRefExpr>(LHSExpr))
        if (const auto *VD_LHS =
                dyn_cast<ValueDecl>(DRE_LHS->getDecl()->getCanonicalDecl());
            VD_LHS && hasOrigin(VD_LHS->getType()))
          addAssignOriginFact(*VD_LHS, *RHSExpr);
    }
  }

private:
  // Check if a type has an origin.
  bool hasOrigin(QualType QT) { return QT->isPointerOrReferenceType(); }

  template <typename Destination, typename Source>
  void addAssignOriginFact(const Destination &D, const Source &S) {
    OriginID DestOID = FactMgr.getOriginMgr().getOrCreate(D);
    OriginID SrcOID = FactMgr.getOriginMgr().get(S);
    CurrentBlockFacts.push_back(
        FactMgr.createFact<AssignOriginFact>(DestOID, SrcOID));
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
        CurrentBlockFacts.push_back(FactMgr.createFact<ExpireFact>(L.ID));
    }
  }

  FactManager &FactMgr;
  AnalysisDeclContext &AC;
  llvm::SmallVector<Fact *> CurrentBlockFacts;
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
    }
    llvm_unreachable("Unknown fact kind");
  }

public:
  Lattice transfer(Lattice In, const IssueFact &) { return In; }
  Lattice transfer(Lattice In, const ExpireFact &) { return In; }
  Lattice transfer(Lattice In, const AssignOriginFact &) { return In; }
  Lattice transfer(Lattice In, const ReturnOfOriginFact &) { return In; }
};

namespace utils {

/// Computes the union of two ImmutableSets.
template <typename T>
llvm::ImmutableSet<T> join(llvm::ImmutableSet<T> A, llvm::ImmutableSet<T> B,
                           typename llvm::ImmutableSet<T>::Factory &F) {
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);
  for (const T &E : B)
    A = F.add(A, E);
  return A;
}

/// Computes the key-wise union of two ImmutableMaps.
// TODO(opt): This key-wise join is a performance bottleneck. A more
// efficient merge could be implemented using a Patricia Trie or HAMT
// instead of the current AVL-tree-based ImmutableMap.
template <typename K, typename V, typename Joiner>
llvm::ImmutableMap<K, V>
join(llvm::ImmutableMap<K, V> A, llvm::ImmutableMap<K, V> B,
     typename llvm::ImmutableMap<K, V>::Factory &F, Joiner joinValues) {
  if (A.getHeight() < B.getHeight())
    std::swap(A, B);

  // For each element in B, join it with the corresponding element in A
  // (or with an empty value if it doesn't exist in A).
  for (const auto &Entry : B) {
    const K &Key = Entry.first;
    const V &ValB = Entry.second;
    if (const V *ValA = A.lookup(Key))
      A = F.add(A, Key, joinValues(*ValA, ValB));
    else
      A = F.add(A, Key, ValB);
  }
  return A;
}
} // namespace utils

// ========================================================================= //
//                          Loan Propagation Analysis
// ========================================================================= //

// Using LLVM's immutable collections is efficient for dataflow analysis
// as it avoids deep copies during state transitions.
// TODO(opt): Consider using a bitset to represent the set of loans.
using LoanSet = llvm::ImmutableSet<LoanID>;
using OriginLoanMap = llvm::ImmutableMap<OriginID, LoanSet>;

/// An object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  OriginLoanMap::Factory OriginMapFactory;
  LoanSet::Factory LoanSetFactory;

  /// Creates a singleton set containing only the given loan ID.
  LoanSet createLoanSet(LoanID LID) {
    return LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID);
  }
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

  LifetimeFactory &Factory;

public:
  LoanPropagationAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                          LifetimeFactory &Factory)
      : DataflowAnalysis(C, AC, F), Factory(Factory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges two lattices by taking the union of loans for each origin.
  // TODO(opt): Keep the state small by removing origins which become dead.
  Lattice join(Lattice A, Lattice B) {
    OriginLoanMap JoinedOrigins =
        utils::join(A.Origins, B.Origins, Factory.OriginMapFactory,
                    [this](LoanSet S1, LoanSet S2) {
                      return utils::join(S1, S2, Factory.LoanSetFactory);
                    });
    return Lattice(JoinedOrigins);
  }

  /// A new loan is issued to the origin. Old loans are erased.
  Lattice transfer(Lattice In, const IssueFact &F) {
    OriginID OID = F.getOriginID();
    LoanID LID = F.getLoanID();
    return LoanPropagationLattice(Factory.OriginMapFactory.add(
        In.Origins, OID, Factory.createLoanSet(LID)));
  }

  /// The destination origin's loan set is replaced by the source's.
  /// This implicitly "resets" the old loans of the destination.
  Lattice transfer(Lattice In, const AssignOriginFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();
    LoanSet SrcLoans = getLoans(In, SrcOID);
    return LoanPropagationLattice(
        Factory.OriginMapFactory.add(In.Origins, DestOID, SrcLoans));
  }

  LoanSet getLoans(OriginID OID, ProgramPoint P) {
    return getLoans(getState(P), OID);
  }

private:
  LoanSet getLoans(Lattice L, OriginID OID) {
    if (auto *Loans = L.Origins.lookup(OID))
      return *Loans;
    return Factory.LoanSetFactory.getEmptySet();
  }
};

// ========================================================================= //
//  TODO:
// - Modify loan expiry analysis to answer `bool isExpired(Loan L, Point P)`
// - Modify origin liveness analysis to answer `bool isLive(Origin O, Point P)`
// - Using the above three to perform the final error reporting.
// ========================================================================= //
} // anonymous namespace

void runLifetimeSafetyAnalysis(const DeclContext &DC, const CFG &Cfg,
                               AnalysisDeclContext &AC) {
  llvm::TimeTraceScope TimeProfile("LifetimeSafetyAnalysis");
  DEBUG_WITH_TYPE("PrintCFG", Cfg.dump(AC.getASTContext().getLangOpts(),
                                       /*ShowColors=*/true));
  FactManager FactMgr;
  FactGenerator FactGen(FactMgr, AC);
  FactGen.run();
  DEBUG_WITH_TYPE("LifetimeFacts", FactMgr.dump(Cfg, AC));

  /// TODO(opt): Consider optimizing individual blocks before running the
  /// dataflow analysis.
  /// 1. Expression Origins: These are assigned once and read at most once,
  ///    forming simple chains. These chains can be compressed into a single
  ///    assignment.
  /// 2. Block-Local Loans: Origins of expressions are never read by other
  ///    blocks; only Decls are visible.  Therefore, loans in a block that
  ///    never reach an Origin associated with a Decl can be safely dropped by
  ///    the analysis.
  LifetimeFactory Factory;
  LoanPropagationAnalysis LoanPropagation(Cfg, AC, FactMgr, Factory);
  LoanPropagation.run();
  DEBUG_WITH_TYPE("LifetimeLoanPropagation", LoanPropagation.dump());
}
} // namespace clang
