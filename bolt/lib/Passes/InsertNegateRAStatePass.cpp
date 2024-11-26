#include "bolt/Passes/InsertNegateRAStatePass.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include <cstdlib>
#include <fstream>
#include <iterator>

using namespace llvm;

namespace llvm {
namespace bolt {

void InsertNegateRAState::runOnFunction(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  if (BF.getState() == BinaryFunction::State::Empty) {
    return;
  }

  if (BF.getState() != BinaryFunction::State::CFG &&
      BF.getState() != BinaryFunction::State::CFG_Finalized) {
    BC.outs() << "BOLT-INFO: No CFG for " << BF.getPrintName()
              << " in InsertNegateRAStatePass\n";
    return;
  }

  if (BF.isIgnored())
    return;

  if (!addNegateRAStateAfterPacOrAuth(BF)) {
    // none inserted, function doesn't need more work
    return;
  }

  auto FirstBB = BF.begin();
  explore_call_graph(BC, &(*FirstBB));

  // We have to do the walk again, starting from any undiscovered autiasp
  // instructions, because some autiasp might not be reachable because of
  // indirect branches but we know that autiasp block should have a Signed
  // state, so we can work out other Unkown states starting from these nodes.
  for (BinaryBasicBlock &BB : BF) {
    if (BBhasAUTH(BC, &BB) && BB.isRAStateUnknown()) {
      BB.setRASigned();
      explore_call_graph(BC, &BB);
    }
  }

  // insert negateRAState-s where there is a State boundary:
  // that is: two consecutive BBs have different RA State
  BinaryFunction::iterator PrevBB;
  bool FirstIter = true;
  for (auto BB = BF.begin(); BB != BF.end(); ++BB) {
    if (!FirstIter) {
      if ((PrevBB->RAState == BinaryBasicBlock::RAStateEnum::Signed &&
           (*BB).RAState == BinaryBasicBlock::RAStateEnum::Unsigned &&
           !BBhasAUTH(BC, &(*PrevBB))) ||
          (PrevBB->RAState == BinaryBasicBlock::RAStateEnum::Signed &&
           (*BB).RAState == BinaryBasicBlock::RAStateEnum::Signed &&
           BBhasAUTH(BC, &(*PrevBB)))) {
        auto InstRevIter = PrevBB->getLastNonPseudo();
        MCInst LastNonPseudo = *InstRevIter;
        auto InstIter = InstRevIter.base();
        BF.addCFIInstruction(&(*PrevBB), InstIter,
                             MCCFIInstruction::createNegateRAState(nullptr));
      }
    } else {
      FirstIter = false;
    }
    PrevBB = BB;
  }
}

void InsertNegateRAState::explore_call_graph(BinaryContext &BC,
                                             BinaryBasicBlock *BB) {
  std::stack<BinaryBasicBlock *> SignedStack;
  std::stack<BinaryBasicBlock *> UnsignedStack;

  // start according to the first BB
  if (BBhasSIGN(BC, BB)) {
    SignedStack.push(BB);
    process_signed_BB(BC, BB, &SignedStack, &UnsignedStack);
  } else {
    UnsignedStack.push(BB);
    process_unsigned_BB(BC, BB, &SignedStack, &UnsignedStack);
  }

  while (!(SignedStack.empty() && UnsignedStack.empty())) {
    if (!SignedStack.empty()) {
      BB = SignedStack.top();
      SignedStack.pop();
      process_signed_BB(BC, BB, &SignedStack, &UnsignedStack);
    } else if (!UnsignedStack.empty()) {
      BB = UnsignedStack.top();
      UnsignedStack.pop();
      process_unsigned_BB(BC, BB, &SignedStack, &UnsignedStack);
    }
  }
}
void InsertNegateRAState::process_signed_BB(
    BinaryContext &BC, BinaryBasicBlock *BB,
    std::stack<BinaryBasicBlock *> *SignedStack,
    std::stack<BinaryBasicBlock *> *UnsignedStack) {

  BB->setRASigned();

  if (BBhasAUTH(BC, BB)) {
    // successors of block with autiasp are stored in the Unsigned Stack
    for (BinaryBasicBlock *Succ : BB->successors()) {
      if (Succ->getFunction() == BB->getFunction() &&
          Succ->isRAStateUnknown()) {
        UnsignedStack->push(Succ);
      }
    }
  } else {
    for (BinaryBasicBlock *Succ : BB->successors()) {
      if (Succ->getFunction() == BB->getFunction() &&
          !Succ->isRAStateSigned()) {
        SignedStack->push(Succ);
      }
    }
  }
  // process predecessors
  if (BBhasSIGN(BC, BB)) {
    for (BinaryBasicBlock *Pred : BB->predecessors()) {
      if (Pred->getFunction() == BB->getFunction() &&
          Pred->isRAStateUnknown()) {
        UnsignedStack->push(Pred);
      }
    }
  } else {
    for (BinaryBasicBlock *Pred : BB->predecessors()) {
      if (Pred->getFunction() == BB->getFunction() &&
          !Pred->isRAStateSigned()) {
        SignedStack->push(Pred);
      }
    }
  }
}

void InsertNegateRAState::process_unsigned_BB(
    BinaryContext &BC, BinaryBasicBlock *BB,
    std::stack<BinaryBasicBlock *> *SignedStack,
    std::stack<BinaryBasicBlock *> *UnsignedStack) {

  BB->setRAUnsigned();

  if (BBhasSIGN(BC, BB)) {
    BB->setRASigned();
    // successors of block with paciasp are stored in the Signed Stack
    for (BinaryBasicBlock *Succ : BB->successors()) {
      if (Succ->getFunction() == BB->getFunction() &&
          !Succ->isRAStateSigned()) {
        SignedStack->push(Succ);
      }
    }
  } else {
    for (BinaryBasicBlock *Succ : BB->successors()) {
      if (Succ->getFunction() == BB->getFunction() &&
          Succ->isRAStateUnknown()) {
        UnsignedStack->push(Succ);
      }
    }
  }

  // process predecessors
  if (BBhasAUTH(BC, BB)) {
    BB->setRASigned();
    for (BinaryBasicBlock *Pred : BB->predecessors()) {
      if (Pred->getFunction() == BB->getFunction() &&
          !Pred->isRAStateSigned()) {
        SignedStack->push(Pred);
      }
    }
  } else {
    for (BinaryBasicBlock *Pred : BB->predecessors()) {
      if (Pred->getFunction() == BB->getFunction() &&
          Pred->isRAStateUnknown()) {
        UnsignedStack->push(Pred);
      }
    }
  }
}

bool InsertNegateRAState::BBhasAUTH(BinaryContext &BC, BinaryBasicBlock *BB) {
  for (auto Iter = BB->begin(); Iter != BB->end(); ++Iter) {
    MCInst Inst = *Iter;
    if (BC.MIB->isPAuth(Inst)) {
      return true;
    }
  }
  return false;
}

bool InsertNegateRAState::BBhasSIGN(BinaryContext &BC, BinaryBasicBlock *BB) {
  for (auto Iter = BB->begin(); Iter != BB->end(); ++Iter) {
    MCInst Inst = *Iter;
    if (BC.MIB->isPSign(Inst)) {
      return true;
    }
  }
  return false;
}

bool InsertNegateRAState::addNegateRAStateAfterPacOrAuth(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  bool FoundAny = false;
  for (BinaryBasicBlock &BB : BF) {
    for (auto Iter = BB.begin(); Iter != BB.end(); ++Iter) {
      MCInst Inst = *Iter;
      if (BC.MIB->isPSign(Inst)) {
        Iter = BF.addCFIInstruction(
            &BB, Iter + 1, MCCFIInstruction::createNegateRAState(nullptr));
        FoundAny = true;
      }

      if (BC.MIB->isPAuth(Inst)) {
        Iter = BF.addCFIInstruction(
            &BB, Iter + 1, MCCFIInstruction::createNegateRAState(nullptr));
        FoundAny = true;
      }
    }
  }
  return FoundAny;
}

Error InsertNegateRAState::runOnFunctions(BinaryContext &BC) {
  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_TRIVIAL, WorkFun, nullptr,
      "InsertNegateRAStatePass");

  return Error::success();
}

} // end namespace bolt
} // end namespace llvm
