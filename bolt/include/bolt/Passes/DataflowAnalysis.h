//===- bolt/Passes/DataflowAnalysis.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_DATAFLOWANALYSIS_H
#define BOLT_PASSES_DATAFLOWANALYSIS_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/Errc.h"
#include <queue>

namespace llvm {
namespace bolt {

/// Represents a given program point as viewed by a dataflow analysis. This
/// point is a location that may be either an instruction or a basic block.
///  Example:
///
///    BB1:    --> ProgramPoint 1  (stored as bb *)
///      add   --> ProgramPoint 2  (stored as inst *)
///      sub   --> ProgramPoint 3  (stored as inst *)
///      jmp   --> ProgramPoint 4  (stored as inst *)
///
/// ProgramPoints allow us to attach a state to any location in the program
/// and is a core concept used in the dataflow analysis engine.
///
/// A dataflow analysis will associate a state with a program point. In
/// analyses whose direction is forward, this state tracks what happened after
/// the execution of an instruction, and the BB tracks the state of what
/// happened before the execution of the first instruction in this BB. For
/// backwards dataflow analyses, state tracks what happened before the
/// execution of a given instruction, while the state associated with a BB
/// tracks what happened after the execution of the last instruction of a BB.
class ProgramPoint {
  enum IDTy : bool { BB = 0, Inst } ID;

  union DataU {
    BinaryBasicBlock *BB;
    MCInst *Inst;
    DataU(BinaryBasicBlock *BB) : BB(BB) {}
    DataU(MCInst *Inst) : Inst(Inst) {}
  } Data;

public:
  ProgramPoint() : ID(IDTy::BB), Data((MCInst *)nullptr) {}
  ProgramPoint(BinaryBasicBlock *BB) : ID(IDTy::BB), Data(BB) {}
  ProgramPoint(MCInst *Inst) : ID(IDTy::Inst), Data(Inst) {}

  /// Convenience function to access the last program point of a basic block,
  /// which is equal to its last instruction. If it is empty, it is equal to
  /// itself.
  static ProgramPoint getLastPointAt(BinaryBasicBlock &BB) {
    auto Last = BB.rbegin();
    if (Last != BB.rend())
      return ProgramPoint(&*Last);
    return ProgramPoint(&BB);
  }

  /// Similar to getLastPointAt.
  static ProgramPoint getFirstPointAt(BinaryBasicBlock &BB) {
    auto First = BB.begin();
    if (First != BB.end())
      return ProgramPoint(&*First);
    return ProgramPoint(&BB);
  }

  bool operator<(const ProgramPoint &PP) const { return Data.BB < PP.Data.BB; }
  bool operator==(const ProgramPoint &PP) const {
    return Data.BB == PP.Data.BB;
  }

  bool isBB() const { return ID == IDTy::BB; }
  bool isInst() const { return ID == IDTy::Inst; }

  BinaryBasicBlock *getBB() const {
    assert(isBB());
    return Data.BB;
  }
  MCInst *getInst() const {
    assert(isInst());
    return Data.Inst;
  }

  friend DenseMapInfo<ProgramPoint>;
};

/// Convenience function to operate on all predecessors of a BB, as viewed
/// by a dataflow analysis. This includes throw sites if it is a landing pad.
void doForAllPreds(const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task);

/// Operates on all successors of a basic block.
void doForAllSuccs(const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task);

/// Default printer for State data.
template <typename StateTy> class StatePrinter {
public:
  void print(raw_ostream &OS, const StateTy &State) const { OS << State; }
  explicit StatePrinter(const BinaryContext &) {}
};

/// Printer for State data that is a BitVector of registers.
class RegStatePrinter {
public:
  void print(raw_ostream &OS, const BitVector &State) const;
  explicit RegStatePrinter(const BinaryContext &BC) : BC(BC) {}

private:
  const BinaryContext &BC;
};

/// Base class for dataflow analyses. Depends on the type of whatever object is
/// stored as the state (StateTy) at each program point. The dataflow then
/// updates the state at each program point depending on the instruction being
/// processed, iterating until all points converge and agree on a state value.
/// Remember that depending on how you formulate your dataflow equation, this
/// may not converge and will loop indefinitely.
/// /p Backward indicates the direction of the dataflow. If false, direction is
/// forward.
///
/// Example: Compute the set of live registers at each program point.
///
///   Modelling:
///     Let State be the set of registers that are live. The kill set of a
///     point is the set of all registers clobbered by the instruction at this
///     program point. The gen set is the set of all registers read by it.
///
///       out{b} = Union (s E succs{b}) {in{s}}
///       in{b}  = (out{b} - kill{b}) U gen{b}
///
///   Template parameters:
///     StateTy = BitVector, where each index corresponds to a machine register
///     Backward = true   (live reg operates in reverse order)
///
///   Subclass implementation notes:
///     Confluence operator = union  (if a reg is alive in any succ, it is alive
///     in the current block).
///
template <typename Derived, typename StateTy, bool Backward = false,
          typename StatePrinterTy = StatePrinter<StateTy>>
class DataflowAnalysis {
  /// CRTP convenience methods
  Derived &derived() { return *static_cast<Derived *>(this); }

  const Derived &const_derived() const {
    return *static_cast<const Derived *>(this);
  }

  mutable Optional<unsigned> AnnotationIndex;

protected:
  const BinaryContext &BC;
  /// Reference to the function being analysed
  BinaryFunction &Func;

  /// The id of the annotation allocator to be used
  MCPlusBuilder::AllocatorIdTy AllocatorId = 0;

  /// Tracks the state at basic block start (end) if direction of the dataflow
  /// is forward (backward).
  std::unordered_map<const BinaryBasicBlock *, StateTy> StateAtBBEntry;
  /// Map a point to its previous (succeeding) point if the direction of the
  /// dataflow is forward (backward). This is used to support convenience
  /// methods to access the resulting state before (after) a given instruction,
  /// otherwise our clients need to keep "prev" pointers themselves.
  DenseMap<const MCInst *, ProgramPoint> PrevPoint;

  /// Perform any bookkeeping before dataflow starts
  void preflight() { llvm_unreachable("Unimplemented method"); }

  /// Sets initial state for each BB
  StateTy getStartingStateAtBB(const BinaryBasicBlock &BB) {
    llvm_unreachable("Unimplemented method");
  }

  /// Sets initial state for each instruction (out set)
  StateTy getStartingStateAtPoint(const MCInst &Point) {
    llvm_unreachable("Unimplemented method");
  }

  /// Computes the in set for the first instruction in a BB by applying the
  /// confluence operator to the out sets of the last instruction of each pred
  /// (in case of a backwards dataflow, we will operate on the in sets of each
  /// successor to determine the starting state of the last instruction of the
  /// current BB)
  void doConfluence(StateTy &StateOut, const StateTy &StateIn) {
    llvm_unreachable("Unimplemented method");
  }

  /// In case of a forwards dataflow, compute the in set for the first
  /// instruction in a Landing Pad considering all out sets for associated
  /// throw sites.
  /// In case of a backwards dataflow, compute the in set of a invoke
  /// instruction considering in sets for the first instructions of its
  /// landing pads.
  void doConfluenceWithLP(StateTy &StateOut, const StateTy &StateIn,
                          const MCInst &Invoke) {
    return derived().doConfluence(StateOut, StateIn);
  }

  /// Returns the out set of an instruction given its in set.
  /// If backwards, computes the in set given its out set.
  StateTy computeNext(const MCInst &Point, const StateTy &Cur) {
    llvm_unreachable("Unimplemented method");
    return StateTy();
  }

  /// Returns the MCAnnotation name
  StringRef getAnnotationName() const {
    llvm_unreachable("Unimplemented method");
    return StringRef("");
  }

  unsigned getAnnotationIndex() const {
    if (AnnotationIndex)
      return *AnnotationIndex;
    AnnotationIndex =
        BC.MIB->getOrCreateAnnotationIndex(const_derived().getAnnotationName());
    return *AnnotationIndex;
  }

  /// Private getter methods accessing state in a read-write fashion
  StateTy &getOrCreateStateAt(const BinaryBasicBlock &BB) {
    return StateAtBBEntry[&BB];
  }

  StateTy &getOrCreateStateAt(MCInst &Point) {
    return BC.MIB->getOrCreateAnnotationAs<StateTy>(
        Point, derived().getAnnotationIndex(), AllocatorId);
  }

  StateTy &getOrCreateStateAt(ProgramPoint Point) {
    if (Point.isBB())
      return getOrCreateStateAt(*Point.getBB());
    return getOrCreateStateAt(*Point.getInst());
  }

public:
  /// Return the allocator id
  unsigned getAllocatorId() { return AllocatorId; }

  /// If the direction of the dataflow is forward, operates on the last
  /// instruction of all predecessors when performing an iteration of the
  /// dataflow equation for the start of this BB.  If backwards, operates on
  /// the first instruction of all successors.
  void doForAllSuccsOrPreds(const BinaryBasicBlock &BB,
                            std::function<void(ProgramPoint)> Task) {
    if (!Backward)
      return doForAllPreds(BB, Task);
    return doForAllSuccs(BB, Task);
  }

  /// We need the current binary context and the function that will be processed
  /// in this dataflow analysis.
  DataflowAnalysis(BinaryFunction &BF,
                   MCPlusBuilder::AllocatorIdTy AllocatorId = 0)
      : BC(BF.getBinaryContext()), Func(BF), AllocatorId(AllocatorId) {}

  virtual ~DataflowAnalysis() { cleanAnnotations(); }

  /// Track the state at basic block start (end) if direction of the dataflow
  /// is forward (backward).
  ErrorOr<const StateTy &> getStateAt(const BinaryBasicBlock &BB) const {
    auto Iter = StateAtBBEntry.find(&BB);
    if (Iter == StateAtBBEntry.end())
      return make_error_code(errc::result_out_of_range);
    return Iter->second;
  }

  /// Track the state at the end (start) of each MCInst in this function if
  /// the direction of the dataflow is forward (backward).
  ErrorOr<const StateTy &> getStateAt(const MCInst &Point) const {
    return BC.MIB->tryGetAnnotationAs<StateTy>(
        Point, const_derived().getAnnotationIndex());
  }

  /// Return the out set (in set) of a given program point if the direction of
  /// the dataflow is forward (backward).
  ErrorOr<const StateTy &> getStateAt(ProgramPoint Point) const {
    if (Point.isBB())
      return getStateAt(*Point.getBB());
    return getStateAt(*Point.getInst());
  }

  /// Relies on a ptr map to fetch the previous instruction and then retrieve
  /// state. WARNING: Watch out for invalidated pointers. Do not use this
  /// function if you invalidated pointers after the analysis has been completed
  ErrorOr<const StateTy &> getStateBefore(const MCInst &Point) {
    return getStateAt(PrevPoint[&Point]);
  }

  ErrorOr<const StateTy &> getStateBefore(ProgramPoint Point) {
    if (Point.isBB())
      return getStateAt(*Point.getBB());
    return getStateAt(PrevPoint[Point.getInst()]);
  }

  /// Remove any state annotations left by this analysis
  void cleanAnnotations() {
    for (BinaryBasicBlock &BB : Func) {
      for (MCInst &Inst : BB) {
        BC.MIB->removeAnnotation(Inst, derived().getAnnotationIndex());
      }
    }
  }

  /// Public entry point that will perform the entire analysis form start to
  /// end.
  void run() {
    derived().preflight();

    if (Func.begin() == Func.end())
      return;
    // Initialize state for all points of the function
    for (BinaryBasicBlock &BB : Func) {
      StateTy &St = getOrCreateStateAt(BB);
      St = derived().getStartingStateAtBB(BB);
      for (MCInst &Inst : BB) {
        StateTy &St = getOrCreateStateAt(Inst);
        St = derived().getStartingStateAtPoint(Inst);
      }
    }

    std::queue<BinaryBasicBlock *> Worklist;
    // TODO: Pushing this in a DFS ordering will greatly speed up the dataflow
    // performance.
    if (!Backward) {
      for (BinaryBasicBlock &BB : Func) {
        Worklist.push(&BB);
        MCInst *Prev = nullptr;
        for (MCInst &Inst : BB) {
          PrevPoint[&Inst] = Prev ? ProgramPoint(Prev) : ProgramPoint(&BB);
          Prev = &Inst;
        }
      }
    } else {
      for (auto I = Func.rbegin(), E = Func.rend(); I != E; ++I) {
        Worklist.push(&*I);
        MCInst *Prev = nullptr;
        for (auto J = (*I).rbegin(), E2 = (*I).rend(); J != E2; ++J) {
          MCInst &Inst = *J;
          PrevPoint[&Inst] = Prev ? ProgramPoint(Prev) : ProgramPoint(&*I);
          Prev = &Inst;
        }
      }
    }

    // Main dataflow loop
    while (!Worklist.empty()) {
      BinaryBasicBlock *BB = Worklist.front();
      Worklist.pop();

      // Calculate state at the entry of first instruction in BB
      StateTy StateAtEntry = getOrCreateStateAt(*BB);
      if (BB->isLandingPad()) {
        doForAllSuccsOrPreds(*BB, [&](ProgramPoint P) {
          if (P.isInst() && BC.MIB->isInvoke(*P.getInst()))
            derived().doConfluenceWithLP(StateAtEntry, *getStateAt(P),
                                         *P.getInst());
          else
            derived().doConfluence(StateAtEntry, *getStateAt(P));
        });
      } else {
        doForAllSuccsOrPreds(*BB, [&](ProgramPoint P) {
          derived().doConfluence(StateAtEntry, *getStateAt(P));
        });
      }

      bool Changed = false;
      StateTy &St = getOrCreateStateAt(*BB);
      if (St != StateAtEntry) {
        Changed = true;
        St = std::move(StateAtEntry);
      }

      // Propagate information from first instruction down to the last one
      StateTy *PrevState = &St;
      const MCInst *LAST = nullptr;
      if (!Backward)
        LAST = &*BB->rbegin();
      else
        LAST = &*BB->begin();

      auto doNext = [&](MCInst &Inst, const BinaryBasicBlock &BB) {
        StateTy CurState = derived().computeNext(Inst, *PrevState);

        if (Backward && BC.MIB->isInvoke(Inst)) {
          BinaryBasicBlock *LBB = Func.getLandingPadBBFor(BB, Inst);
          if (LBB) {
            auto First = LBB->begin();
            if (First != LBB->end())
              derived().doConfluenceWithLP(CurState,
                                           getOrCreateStateAt(&*First), Inst);
            else
              derived().doConfluenceWithLP(CurState, getOrCreateStateAt(LBB),
                                           Inst);
          }
        }

        StateTy &St = getOrCreateStateAt(Inst);
        if (St != CurState) {
          St = CurState;
          if (&Inst == LAST)
            Changed = true;
        }
        PrevState = &St;
      };

      if (!Backward)
        for (MCInst &Inst : *BB)
          doNext(Inst, *BB);
      else
        for (auto I = BB->rbegin(), E = BB->rend(); I != E; ++I)
          doNext(*I, *BB);

      if (Changed) {
        if (!Backward) {
          for (BinaryBasicBlock *Succ : BB->successors())
            Worklist.push(Succ);
          for (BinaryBasicBlock *LandingPad : BB->landing_pads())
            Worklist.push(LandingPad);
        } else {
          for (BinaryBasicBlock *Pred : BB->predecessors())
            Worklist.push(Pred);
          for (BinaryBasicBlock *Thrower : BB->throwers())
            Worklist.push(Thrower);
        }
      }
    } // end while (!Worklist.empty())
  }
};

/// Define an iterator for navigating the expressions calculated by a
/// dataflow analysis at each program point, when they are backed by a
/// BitVector.
class ExprIterator {
  const BitVector *BV;
  const std::vector<MCInst *> &Expressions;
  int Idx;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = const MCInst *;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  ExprIterator &operator++() {
    assert(Idx != -1 && "Iterator already at the end");
    Idx = BV->find_next(Idx);
    return *this;
  }
  ExprIterator operator++(int) {
    assert(Idx != -1 && "Iterator already at the end");
    ExprIterator Ret = *this;
    ++(*this);
    return Ret;
  }
  bool operator==(const ExprIterator &Other) const { return Idx == Other.Idx; }
  bool operator!=(const ExprIterator &Other) const { return Idx != Other.Idx; }
  MCInst *operator*() {
    assert(Idx != -1 && "Invalid access to end iterator");
    return Expressions[Idx];
  }
  ExprIterator(const BitVector *BV, const std::vector<MCInst *> &Exprs)
      : BV(BV), Expressions(Exprs) {
    Idx = BV->find_first();
  }
  ExprIterator(const BitVector *BV, const std::vector<MCInst *> &Exprs, int Idx)
      : BV(BV), Expressions(Exprs), Idx(Idx) {}

  int getBitVectorIndex() const { return Idx; }
};

/// Specialization of DataflowAnalysis whose state specifically stores
/// a set of instructions.
template <typename Derived, bool Backward = false,
          typename StatePrinterTy = StatePrinter<BitVector>>
class InstrsDataflowAnalysis
    : public DataflowAnalysis<Derived, BitVector, Backward, StatePrinterTy> {
public:
  /// These iterator functions offer access to the set of pointers to
  /// instructions in a given program point
  template <typename T> ExprIterator expr_begin(const T &Point) const {
    if (auto State = this->getStateAt(Point))
      return ExprIterator(&*State, Expressions);
    return expr_end();
  }
  ExprIterator expr_begin(const BitVector &BV) const {
    return ExprIterator(&BV, Expressions);
  }
  ExprIterator expr_end() const {
    return ExprIterator(nullptr, Expressions, -1);
  }

  /// Used to size the set of expressions/definitions being tracked by the
  /// dataflow analysis
  uint64_t NumInstrs{0};
  /// We put every MCInst we want to track (which one representing an
  /// expression/def) into a vector because we need to associate them with
  /// small numbers. They will be tracked via BitVectors throughout the
  /// dataflow analysis.
  std::vector<MCInst *> Expressions;
  /// Maps expressions defs (MCInsts) to its index in the Expressions vector
  std::unordered_map<const MCInst *, uint64_t> ExprToIdx;

  /// Return whether \p Expr is in the state set at \p Point
  bool count(ProgramPoint Point, const MCInst &Expr) const {
    auto IdxIter = ExprToIdx.find(&Expr);
    assert(IdxIter != ExprToIdx.end() && "Invalid Expr");
    return (*this->getStateAt(Point))[IdxIter->second];
  }

  bool count(const MCInst &Point, const MCInst &Expr) const {
    auto IdxIter = ExprToIdx.find(&Expr);
    assert(IdxIter != ExprToIdx.end() && "Invalid Expr");
    return (*this->getStateAt(Point))[IdxIter->second];
  }

  /// Return whether \p Expr is in the state set at the instr of index
  /// \p PointIdx
  bool count(unsigned PointIdx, const MCInst &Expr) const {
    return count(*Expressions[PointIdx], Expr);
  }

  InstrsDataflowAnalysis(BinaryFunction &BF,
                         MCPlusBuilder::AllocatorIdTy AllocId = 0)
      : DataflowAnalysis<Derived, BitVector, Backward, StatePrinterTy>(
            BF, AllocId) {}
  virtual ~InstrsDataflowAnalysis() {}
};

} // namespace bolt

/// DenseMapInfo allows us to use the DenseMap LLVM data structure to store
/// ProgramPoints.
template <> struct DenseMapInfo<bolt::ProgramPoint> {
  static inline bolt::ProgramPoint getEmptyKey() {
    uintptr_t Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<MCInst *>::NumLowBitsAvailable;
    return bolt::ProgramPoint(reinterpret_cast<MCInst *>(Val));
  }
  static inline bolt::ProgramPoint getTombstoneKey() {
    uintptr_t Val = static_cast<uintptr_t>(-2);
    Val <<= PointerLikeTypeTraits<MCInst *>::NumLowBitsAvailable;
    return bolt::ProgramPoint(reinterpret_cast<MCInst *>(Val));
  }
  static unsigned getHashValue(const bolt::ProgramPoint &PP) {
    return (unsigned((uintptr_t)PP.Data.BB) >> 4) ^
           (unsigned((uintptr_t)PP.Data.BB) >> 9);
  }
  static bool isEqual(const bolt::ProgramPoint &LHS,
                      const bolt::ProgramPoint &RHS) {
    return LHS.Data.BB == RHS.Data.BB;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const BitVector &Val);

} // namespace llvm

#endif
