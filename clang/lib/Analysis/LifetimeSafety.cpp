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
//                              The Dataflow Lattice
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
  LoanSet::Factory LoanSetFact;

  /// Creates a singleton set containing only the given loan ID.
  LoanSet createLoanSet(LoanID LID) {
    return LoanSetFact.add(LoanSetFact.getEmptySet(), LID);
  }
};

/// LifetimeLattice represents the state of our analysis at a given program
/// point. It is an immutable object, and all operations produce a new
/// instance rather than modifying the existing one.
struct LifetimeLattice {
  /// The map from an origin to the set of loans it contains.
  /// The lattice has a finite height: An origin's loan set is bounded by the
  /// total number of loans in the function.
  /// TODO(opt): To reduce the lattice size, propagate origins of declarations,
  /// not expressions, because expressions are not visible across blocks.
  OriginLoanMap Origins = OriginLoanMap(nullptr);

  explicit LifetimeLattice(const OriginLoanMap &S) : Origins(S) {}
  LifetimeLattice() = default;

  bool operator==(const LifetimeLattice &Other) const {
    return Origins == Other.Origins;
  }
  bool operator!=(const LifetimeLattice &Other) const {
    return !(*this == Other);
  }

  LoanSet getLoans(OriginID OID) const {
    if (auto *Loans = Origins.lookup(OID))
      return *Loans;
    return LoanSet(nullptr);
  }

  /// Computes the union of two lattices by performing a key-wise join of
  /// their OriginLoanMaps.
  // TODO(opt): This key-wise join is a performance bottleneck. A more
  // efficient merge could be implemented using a Patricia Trie or HAMT
  // instead of the current AVL-tree-based ImmutableMap.
  // TODO(opt): Keep the state small by removing origins which become dead.
  LifetimeLattice join(const LifetimeLattice &Other,
                       LifetimeFactory &Factory) const {
    /// Merge the smaller map into the larger one ensuring we iterate over the
    /// smaller map.
    if (Origins.getHeight() < Other.Origins.getHeight())
      return Other.join(*this, Factory);

    OriginLoanMap JoinedState = Origins;
    // For each origin in the other map, union its loan set with ours.
    for (const auto &Entry : Other.Origins) {
      OriginID OID = Entry.first;
      LoanSet OtherLoanSet = Entry.second;
      JoinedState = Factory.OriginMapFactory.add(
          JoinedState, OID, join(getLoans(OID), OtherLoanSet, Factory));
    }
    return LifetimeLattice(JoinedState);
  }

  LoanSet join(LoanSet a, LoanSet b, LifetimeFactory &Factory) const {
    /// Merge the smaller set into the larger one ensuring we iterate over the
    /// smaller set.
    if (a.getHeight() < b.getHeight())
      std::swap(a, b);
    LoanSet Result = a;
    for (LoanID LID : b) {
      /// TODO(opt): Profiling shows that this loop is a major performance
      /// bottleneck. Investigate using a BitVector to represent the set of
      /// loans for improved join performance.
      Result = Factory.LoanSetFact.add(Result, LID);
    }
    return Result;
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "LifetimeLattice State:\n";
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

// ========================================================================= //
//                              The Transfer Function
// ========================================================================= //
class Transferer {
  FactManager &AllFacts;
  LifetimeFactory &Factory;

public:
  explicit Transferer(FactManager &F, LifetimeFactory &Factory)
      : AllFacts(F), Factory(Factory) {}

  /// Computes the exit state of a block by applying all its facts sequentially
  /// to a given entry state.
  /// TODO: We might need to store intermediate states per-fact in the block for
  /// later analysis.
  LifetimeLattice transferBlock(const CFGBlock *Block,
                                LifetimeLattice EntryState) {
    LifetimeLattice BlockState = EntryState;
    llvm::ArrayRef<const Fact *> Facts = AllFacts.getFacts(Block);

    for (const Fact *F : Facts) {
      BlockState = transferFact(BlockState, F);
    }
    return BlockState;
  }

private:
  LifetimeLattice transferFact(LifetimeLattice In, const Fact *F) {
    switch (F->getKind()) {
    case Fact::Kind::Issue:
      return transfer(In, *F->getAs<IssueFact>());
    case Fact::Kind::AssignOrigin:
      return transfer(In, *F->getAs<AssignOriginFact>());
    // Expire and ReturnOfOrigin facts don't modify the Origins and the State.
    case Fact::Kind::Expire:
    case Fact::Kind::ReturnOfOrigin:
      return In;
    }
    llvm_unreachable("Unknown fact kind");
  }

  /// A new loan is issued to the origin. Old loans are erased.
  LifetimeLattice transfer(LifetimeLattice In, const IssueFact &F) {
    OriginID OID = F.getOriginID();
    LoanID LID = F.getLoanID();
    return LifetimeLattice(Factory.OriginMapFactory.add(
        In.Origins, OID, Factory.createLoanSet(LID)));
  }

  /// The destination origin's loan set is replaced by the source's.
  /// This implicitly "resets" the old loans of the destination.
  LifetimeLattice transfer(LifetimeLattice InState, const AssignOriginFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();
    LoanSet SrcLoans = InState.getLoans(SrcOID);
    return LifetimeLattice(
        Factory.OriginMapFactory.add(InState.Origins, DestOID, SrcLoans));
  }
};

// ========================================================================= //
//                              Dataflow analysis
// ========================================================================= //

/// Drives the intra-procedural dataflow analysis.
///
/// Orchestrates the analysis by iterating over the CFG using a worklist
/// algorithm. It computes a fixed point by propagating the LifetimeLattice
/// state through each block until the state no longer changes.
/// TODO: Maybe use the dataflow framework! The framework might need changes
/// to support the current comparison done at block-entry.
class LifetimeDataflow {
  const CFG &Cfg;
  AnalysisDeclContext &AC;
  LifetimeFactory LifetimeFact;

  Transferer Xfer;

  /// Stores the merged analysis state at the entry of each CFG block.
  llvm::DenseMap<const CFGBlock *, LifetimeLattice> BlockEntryStates;
  /// Stores the analysis state at the exit of each CFG block, after the
  /// transfer function has been applied.
  llvm::DenseMap<const CFGBlock *, LifetimeLattice> BlockExitStates;

public:
  LifetimeDataflow(const CFG &C, FactManager &FS, AnalysisDeclContext &AC)
      : Cfg(C), AC(AC), Xfer(FS, LifetimeFact) {}

  void run() {
    llvm::TimeTraceScope TimeProfile("Lifetime Dataflow");
    ForwardDataflowWorklist Worklist(Cfg, AC);
    const CFGBlock *Entry = &Cfg.getEntry();
    BlockEntryStates[Entry] = LifetimeLattice{};
    Worklist.enqueueBlock(Entry);
    while (const CFGBlock *B = Worklist.dequeue()) {
      LifetimeLattice EntryState = getEntryState(B);
      LifetimeLattice ExitState = Xfer.transferBlock(B, EntryState);
      BlockExitStates[B] = ExitState;

      for (const CFGBlock *Successor : B->succs()) {
        auto SuccIt = BlockEntryStates.find(Successor);
        LifetimeLattice OldSuccEntryState = (SuccIt != BlockEntryStates.end())
                                                ? SuccIt->second
                                                : LifetimeLattice{};
        LifetimeLattice NewSuccEntryState =
            OldSuccEntryState.join(ExitState, LifetimeFact);
        // Enqueue the successor if its entry state has changed.
        // TODO(opt): Consider changing 'join' to report a change if !=
        // comparison is found expensive.
        if (SuccIt == BlockEntryStates.end() ||
            NewSuccEntryState != OldSuccEntryState) {
          BlockEntryStates[Successor] = NewSuccEntryState;
          Worklist.enqueueBlock(Successor);
        }
      }
    }
  }

  void dump() const {
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << "       Dataflow results:\n";
    llvm::dbgs() << "==========================================\n";
    const CFGBlock &B = Cfg.getExit();
    getExitState(&B).dump(llvm::dbgs());
  }

  LifetimeLattice getEntryState(const CFGBlock *B) const {
    return BlockEntryStates.lookup(B);
  }

  LifetimeLattice getExitState(const CFGBlock *B) const {
    return BlockExitStates.lookup(B);
  }
};

// ========================================================================= //
//  TODO: Analysing dataflow results and error reporting.
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
  LifetimeDataflow Dataflow(Cfg, FactMgr, AC);
  Dataflow.run();
  DEBUG_WITH_TYPE("LifetimeDataflow", Dataflow.dump());
}
} // namespace clang
