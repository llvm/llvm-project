#include "clang/Analysis/Analyses/LifetimeSafety.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowWorklist.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TimeProfiler.h"
#include <vector>

namespace clang {
namespace {

struct Point {
  const clang::CFGBlock *Block;
  /// Index into Block->Elements().
  unsigned ElementIndex;

  Point(const clang::CFGBlock *B = nullptr, unsigned Idx = 0)
      : Block(B), ElementIndex(Idx) {}

  bool operator==(const Point &Other) const {
    return Block == Other.Block && ElementIndex == Other.ElementIndex;
  }
};

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable.
/// TODO: Handle member accesseslike `s.y`.
struct Path {
  const clang::ValueDecl *D;

  enum class Kind : uint8_t {
    StackVariable,
    Heap,            // TODO: Handle.
    Field,           // TODO: Handle.
    ArrayElement,    // TODO: Handle.
    TemporaryObject, // TODO: Handle.
    StaticOrGlobal,  // TODO: Handle.
  };

  Kind PathKind;

  Path(const clang::ValueDecl *D, Kind K) : D(D), PathKind(K) {}
};

using LoanID = uint32_t;
using OriginID = uint32_t;

/// Information about a single borrow, or "Loan". A loan is created when a
/// reference or pointer is taken.
struct LoanInfo {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
  LoanID ID;
  Path SourcePath;
  SourceLocation IssueLoc;

  LoanInfo(LoanID id, Path path, SourceLocation loc)
      : ID(id), SourcePath(path), IssueLoc(loc) {}
};

enum class OriginKind : uint8_t { Variable, ExpressionResult };

/// An Origin is a symbolic identifier that represents the set of possible
/// loans a pointer-like object could hold at any given time.
/// TODO: Also represent Origins of complex types (fields, inner types).
struct OriginInfo {
  OriginID ID;
  OriginKind Kind;
  union {
    const clang::ValueDecl *Decl;
    const clang::Expr *Expression;
  };
  OriginInfo(OriginID id, OriginKind kind, const clang::ValueDecl *D)
      : ID(id), Kind(kind), Decl(D) {}
  OriginInfo(OriginID id, OriginKind kind, const clang::Expr *E)
      : ID(id), Kind(kind), Expression(E) {}
};

class LoanManager {
public:
  LoanManager() = default;

  LoanInfo &addLoanInfo(Path path, SourceLocation loc) {
    NextLoanIDVal++;
    AllLoans.emplace_back(NextLoanIDVal, path, loc);
    return AllLoans.back();
  }

  const LoanInfo *getLoanInfo(LoanID id) const {
    if (id < AllLoans.size())
      return &AllLoans[id];
    return nullptr;
  }
  llvm::ArrayRef<LoanInfo> getLoanInfos() const { return AllLoans; }

private:
  LoanID NextLoanIDVal = 0;
  llvm::SmallVector<LoanInfo> AllLoans;
};

class OriginManager {
public:
  OriginManager() = default;

  OriginID getNextOriginID() { return NextOriginIDVal++; }
  OriginInfo &addOriginInfo(OriginID id, const clang::ValueDecl *D) {
    assert(D != nullptr);
    AllOrigins.emplace_back(id, OriginKind::Variable, D);
    return AllOrigins.back();
  }
  OriginInfo &addOriginInfo(OriginID id, const clang::Expr *E) {
    assert(E != nullptr);
    AllOrigins.emplace_back(id, OriginKind::ExpressionResult, E);
    return AllOrigins.back();
  }

  OriginID getOrCreate(const Expr *E) {
    auto It = ExprToOriginID.find(E);
    if (It != ExprToOriginID.end())
      return It->second;

    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      // Origin of DeclRefExpr is that of the declaration it refers to.
      return getOrCreate(DRE->getDecl());
    }
    OriginID NewID = getNextOriginID();
    addOriginInfo(NewID, E);
    ExprToOriginID[E] = NewID;
    return NewID;
  }

  const OriginInfo *getOriginInfo(OriginID id) const {
    if (id < AllOrigins.size())
      return &AllOrigins[id];
    return nullptr;
  }

  llvm::ArrayRef<OriginInfo> getOriginInfos() const { return AllOrigins; }

  OriginID getOrCreate(const ValueDecl *D) {
    auto It = DeclToOriginID.find(D);
    if (It != DeclToOriginID.end())
      return It->second;
    OriginID NewID = getNextOriginID();
    addOriginInfo(NewID, D);
    DeclToOriginID[D] = NewID;
    return NewID;
  }

private:
  OriginID NextOriginIDVal = 0;
  llvm::SmallVector<OriginInfo> AllOrigins;
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
    /// An origin is part of a function's return value.
    ReturnOfOrigin
  };

private:
  Kind K;

protected:
  Point P;
  Fact(Kind K, Point Pt) : K(K), P(Pt) {}

public:
  virtual ~Fact() = default;
  Kind getKind() const { return K; }
  Point getPoint() const { return P; }

  template <typename T> const T *getAs() const {
    if (T::classof(this))
      return static_cast<const T *>(this);
    return nullptr;
  }

  virtual void dump(llvm::raw_ostream &OS) const {
    OS << "Fact (Kind: " << static_cast<int>(K) << ", Point: B"
       << P.Block->getBlockID() << ":" << P.ElementIndex << ")\n";
  }
};

class IssueFact : public Fact {
  LoanID LID;
  OriginID OID;

public:
  IssueFact(LoanID LID, OriginID OID, Point Pt)
      : Fact(Kind::Issue, Pt), LID(LID), OID(OID) {}
  LoanID getLoanID() const { return LID; }
  OriginID getOriginID() const { return OID; }
  static bool classof(const Fact *F) { return F->getKind() == Kind::Issue; }
  void dump(llvm::raw_ostream &OS) const override {
    OS << "Issue (LoanID: " << getLoanID() << ", OriginID: " << getOriginID()
       << ")\n";
  }
};

class ExpireFact : public Fact {
  LoanID LID;

public:
  ExpireFact(LoanID LID, Point Pt) : Fact(Kind::Expire, Pt), LID(LID) {}
  LoanID getLoanID() const { return LID; }
  static bool classof(const Fact *F) { return F->getKind() == Kind::Expire; }
  void dump(llvm::raw_ostream &OS) const override {
    OS << "Expire (LoanID: " << getLoanID() << ")\n";
  }
};

class AssignOriginFact : public Fact {
  OriginID OIDDest;
  OriginID OIDSrc;

public:
  AssignOriginFact(OriginID OIDDest, OriginID OIDSrc, Point Pt)
      : Fact(Kind::AssignOrigin, Pt), OIDDest(OIDDest), OIDSrc(OIDSrc) {}
  OriginID getDestOriginID() const { return OIDDest; }
  OriginID getSrcOriginID() const { return OIDSrc; }
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::AssignOrigin;
  }
  void dump(llvm::raw_ostream &OS) const override {
    OS << "AssignOrigin (DestID: " << getDestOriginID()
       << ", SrcID: " << getSrcOriginID() << ")\n";
  }
};

class ReturnOfOriginFact : public Fact {
  OriginID OID;

public:
  ReturnOfOriginFact(OriginID OID, Point Pt)
      : Fact(Kind::ReturnOfOrigin, Pt), OID(OID) {}
  OriginID getReturnedOriginID() const { return OID; }
  static bool classof(const Fact *F) {
    return F->getKind() == Kind::ReturnOfOrigin;
  }
  void dump(llvm::raw_ostream &OS) const override {
    OS << "ReturnOfOrigin (OriginID: " << getReturnedOriginID() << ")\n";
  }
};

class FactManager {
public:
  llvm::ArrayRef<Fact *> getFacts(const CFGBlock *B) const {
    auto It = BlockToFactsMap.find(B);
    if (It != BlockToFactsMap.end())
      return It->second;
    return {};
  }

  void addFact(const CFGBlock *B, Fact *NewFact) {
    BlockToFactsMap[B].push_back(NewFact);
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
    if (const Decl *D = AC.getDecl()) {
      if (const auto *ND = dyn_cast<NamedDecl>(D))
        llvm::dbgs() << "Function: " << ND->getQualifiedNameAsString() << "\n";
    }
    // Print blocks in the order as they appear in code for a stable ordering.
    ForwardDataflowWorklist worklist(Cfg, AC);
    for (const CFGBlock *B : Cfg.const_nodes())
      worklist.enqueueBlock(B);
    while (const CFGBlock *B = worklist.dequeue()) {
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

private:
  llvm::DenseMap<const clang::CFGBlock *, llvm::SmallVector<Fact *>>
      BlockToFactsMap;
  llvm::BumpPtrAllocator FactAllocator;
};

struct FactsContext {
  FactManager Facts;
  LoanManager Loans;
  OriginManager Origins;
};

class FactGenerator : public ConstStmtVisitor<FactGenerator> {

public:
  FactGenerator(const CFG &Cfg, FactsContext &FactsCtx, AnalysisDeclContext &AC)
      : FactsCtx(FactsCtx), Cfg(Cfg), AC(AC), CurrentBlock(nullptr) {}

  void run() {
    llvm::TimeTraceScope TimeProfile("Fact Generation");
    // Iterate through the CFG blocks in pre-order to ensure that
    // initializations and destructions are processed in the correct sequence.
    // TODO: A pre-order traversal utility should be provided by Dataflow
    // framework.
    ForwardDataflowWorklist Worklist(Cfg, AC);
    for (const CFGBlock *B : Cfg.const_nodes())
      Worklist.enqueueBlock(B);
    while (const CFGBlock *Block = Worklist.dequeue()) {
      CurrentBlock = Block;
      for (unsigned I = 0; I < Block->size(); ++I) {
        const CFGElement &Element = Block->Elements[I];
        CurrentPoint = Point(Block, I);
        if (std::optional<CFGStmt> CS = Element.getAs<CFGStmt>())
          Visit(CS->getStmt());
        else if (std::optional<CFGAutomaticObjDtor> DtorOpt =
                     Element.getAs<CFGAutomaticObjDtor>())
          handleDestructor(*DtorOpt);
      }
    }
  }

  void VisitDeclStmt(const DeclStmt *DS) {
    for (const Decl *D : DS->decls()) {
      if (const auto *VD = dyn_cast<VarDecl>(D)) {
        if (hasOrigin(VD->getType())) {
          if (const Expr *InitExpr = VD->getInit()) {
            OriginID DestOID = FactsCtx.Origins.getOrCreate(VD);
            OriginID SrcOID = FactsCtx.Origins.getOrCreate(InitExpr);
            FactsCtx.Facts.addFact(CurrentBlock,
                                   FactsCtx.Facts.createFact<AssignOriginFact>(
                                       DestOID, SrcOID, CurrentPoint));
          }
        }
      }
    }
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *ICE) {
    if (!hasOrigin(ICE->getType()))
      return;
    // An ImplicitCastExpr node itself gets an origin, which flows from the
    // origin of its sub-expression (after stripping its own parens/casts).
    if (ICE->getCastKind() == CK_LValueToRValue) {
      OriginID IceOID = FactsCtx.Origins.getOrCreate(ICE);
      OriginID SubExprOID = FactsCtx.Origins.getOrCreate(ICE->getSubExpr());
      FactsCtx.Facts.addFact(CurrentBlock,
                             FactsCtx.Facts.createFact<AssignOriginFact>(
                                 IceOID, SubExprOID, CurrentPoint));
    }
  }

  void VisitUnaryOperator(const UnaryOperator *UO) {
    if (UO->getOpcode() == UO_AddrOf) {
      const Expr *SubExpr = UO->getSubExpr();
      if (const auto *DRE = dyn_cast<DeclRefExpr>(SubExpr)) {
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          // Check if it's a local variable.
          if (VD->hasLocalStorage()) {
            OriginID OID = FactsCtx.Origins.getOrCreate(UO);
            Path AddrOfLocalVarPath(VD, Path::Kind::StackVariable);
            LoanInfo &Loan = FactsCtx.Loans.addLoanInfo(AddrOfLocalVarPath,
                                                        UO->getOperatorLoc());
            FactsCtx.Facts.addFact(CurrentBlock,
                                   FactsCtx.Facts.createFact<IssueFact>(
                                       Loan.ID, OID, CurrentPoint));
          }
        }
      }
    }
  }

  void VisitReturnStmt(const ReturnStmt *RS) {
    if (const Expr *RetExpr = RS->getRetValue()) {
      if (hasOrigin(RetExpr->getType())) {
        OriginID OID = FactsCtx.Origins.getOrCreate(RetExpr);
        FactsCtx.Facts.addFact(
            CurrentBlock,
            FactsCtx.Facts.createFact<ReturnOfOriginFact>(OID, CurrentPoint));
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
      if (const auto *DRE_LHS = dyn_cast<DeclRefExpr>(LHSExpr)) {
        if (const auto *VD_LHS =
                dyn_cast<ValueDecl>(DRE_LHS->getDecl()->getCanonicalDecl());
            VD_LHS && hasOrigin(VD_LHS->getType())) {
          OriginID DestOID = FactsCtx.Origins.getOrCreate(VD_LHS);
          OriginID SrcOID = FactsCtx.Origins.getOrCreate(RHSExpr);
          FactsCtx.Facts.addFact(CurrentBlock,
                                 FactsCtx.Facts.createFact<AssignOriginFact>(
                                     DestOID, SrcOID, CurrentPoint));
        }
      }
    }
  }

private:
  // Check if a type have an origin.
  bool hasOrigin(QualType QT) { return QT->isPointerOrReferenceType(); }

  void handleDestructor(const CFGAutomaticObjDtor &DtorOpt) {
    /// TODO: Also handle trivial destructors (e.g., for `int`
    // variables) which will never have a CFGAutomaticObjDtor node.
    /// TODO: Handle loans to temporaries.
    const VarDecl *DestructedVD = DtorOpt.getVarDecl();
    if (!DestructedVD)
      return;
    // Iterate through all loans to see if any expire.
    for (const LoanInfo &Loan : FactsCtx.Loans.getLoanInfos()) {
      const Path &LoanPath = Loan.SourcePath;
      // Check if the loan is for a stack variable and if that variable
      // is the one being destructed.
      if (LoanPath.PathKind == Path::Kind::StackVariable) {
        if (LoanPath.D == DestructedVD) {
          FactsCtx.Facts.addFact(
              CurrentBlock,
              FactsCtx.Facts.createFact<ExpireFact>(Loan.ID, CurrentPoint));
        }
      }
    }
  }

  FactsContext &FactsCtx;
  const CFG &Cfg;
  AnalysisDeclContext &AC;
  const CFGBlock *CurrentBlock;
  Point CurrentPoint;
};

// ========================================================================= //
//                              The Dataflow Lattice
// ========================================================================= //

// Using LLVM's immutable collections is efficient for dataflow analysis
// as it avoids deep copies during state transitions.
// TODO(opt): Consider using a bitset to represent the set of loans.
using LoanSet = llvm::ImmutableSet<LoanID>;
using OriginLoanMap = llvm::ImmutableMap<OriginID, LoanSet>;

/// A context object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  OriginLoanMap::Factory OriginMapFact;
  LoanSet::Factory LoanSetFact;

  LoanSet createLoanSet(LoanID LID) {
    return LoanSetFact.add(LoanSetFact.getEmptySet(), LID);
  }
};

/// LifetimeLattice represents the state of our analysis at a given program
/// point. It is an immutable object, and all operations produce a new
/// instance rather than modifying the existing one.
struct LifetimeLattice {
  /// The map from an origin to the set of loans it contains.
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

  LoanSet getLoans(OriginID OID, LifetimeFactory &Factory) const {
    if (auto *Loans = Origins.lookup(OID))
      return *Loans;
    return Factory.LoanSetFact.getEmptySet();
  }

  /// Computes the union of two lattices by performing a key-wise merge of
  /// their OriginLoanMaps.
  LifetimeLattice merge(const LifetimeLattice &Other,
                        LifetimeFactory &Factory) const {
    /// Merge the smaller map into the larger one ensuring we iterate over the
    /// smaller map.
    if (Origins.getHeight() < Other.Origins.getHeight())
      return Other.merge(*this, Factory);

    OriginLoanMap MergedState = Origins;
    // For each origin in the other map, union its loan set with ours.
    for (const auto &Entry : Other.Origins) {
      OriginID OID = Entry.first;
      LoanSet OtherLoanSet = Entry.second;
      MergedState = Factory.OriginMapFact.add(
          MergedState, OID,
          merge(getLoans(OID, Factory), OtherLoanSet, Factory));
    }
    return LifetimeLattice(MergedState);
  }

  LoanSet merge(LoanSet a, LoanSet b, LifetimeFactory &Factory) const {
    /// Merge the smaller set into the larger one ensuring we iterate over the
    /// smaller set.
    if (a.getHeight() < b.getHeight())
      std::swap(a, b);
    LoanSet Result = a;
    for (LoanID LID : b) {
      /// TODO(opt): Profiling shows that this loop is a major performance
      /// bottleneck. Investigate using a BitVector to represent the set of
      /// loans for improved merge performance.
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
    llvm::ArrayRef<Fact *> Facts = AllFacts.getFacts(Block);

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
    return LifetimeLattice(
        Factory.OriginMapFact.add(In.Origins, OID, Factory.createLoanSet(LID)));
  }

  /// The destination origin's loan set is replaced by the source's.
  /// This implicitly "resets" the old loans of the destination.
  LifetimeLattice transfer(LifetimeLattice InState, const AssignOriginFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();
    LoanSet SrcLoans = InState.getLoans(SrcOID, Factory);
    return LifetimeLattice(
        Factory.OriginMapFact.add(InState.Origins, DestOID, SrcLoans));
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
/// TODO: Maybe use the dataflow framework!
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
            OldSuccEntryState.merge(ExitState, LifetimeFact);
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
    auto It = BlockEntryStates.find(B);
    if (It != BlockEntryStates.end()) {
      return It->second;
    }
    return LifetimeLattice{};
  }

  LifetimeLattice getExitState(const CFGBlock *B) const {
    auto It = BlockExitStates.find(B);
    if (It != BlockExitStates.end()) {
      return It->second;
    }
    return LifetimeLattice{};
  }
};

// ========================================================================= //
//  TODO: Analysing dataflow results and error reporting.
// ========================================================================= //
} // anonymous namespace

void runLifetimeAnalysis(const DeclContext &DC, const CFG &Cfg,
                         AnalysisDeclContext &AC) {
  llvm::TimeTraceScope TimeProfile("Lifetime Analysis");
  FactsContext FactsCtx;
  FactGenerator FactGen(Cfg, FactsCtx, AC);
  FactGen.run();
  DEBUG_WITH_TYPE("LifetimeFacts", FactsCtx.Facts.dump(Cfg, AC));

  /// TODO(opt): Consider optimizing individual blocks before running the
  /// dataflow analysis.
  /// 1. Expression Origins: These are assigned once and read at most once,
  ///    forming simple chains. These chains can be compressed into a single
  ///    assignment.
  /// 2. Block-Local Loans: Origins of expressions are never read by other
  ///    blocks; only Decls are visible.  Therefore, loans in a block that
  ///    never reach an Origin associated with a Decl can be safely dropped by
  ///    the analysis.
  LifetimeDataflow Dataflow(Cfg, FactsCtx.Facts, AC);
  Dataflow.run();
  DEBUG_WITH_TYPE("LifetimeDataflow", Dataflow.dump());
}
} // namespace clang