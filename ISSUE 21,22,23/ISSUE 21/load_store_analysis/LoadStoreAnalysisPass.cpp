#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
	// Add this before the LoadStoreAnalysisPass struct definition
	struct MemoryAccessInfo {
		enum AccessPattern {
		  DirectLoad,
		  DirectStore,
		  PointerArithmetic,
		  NestedStructures,
		  Unknown
		};

		AccessPattern pattern;
		Instruction *inst;

		MemoryAccessInfo(AccessPattern pattern, Instruction *inst)
		    : pattern(pattern), inst(inst) {}
	};
  // Define the pass class
  struct LoadStoreAnalysisPass : public FunctionPass {
    static char ID;
    
    LoadStoreAnalysisPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
			errs() << "Function: " << F.getName() << '\n';

			for (auto &BB : F) {
				for (auto &I : BB) {
				  if (I.getOpcode() == Instruction::Load) {
				    errs() << "Found a load instruction: " << I << '\n';
				    MemoryAccessInfo info(MemoryAccessInfo::DirectLoad, &I);
				    // TODO: Process the load instruction with the access info
				  } else if (I.getOpcode() == Instruction::Store) {
				    errs() << "Found a store instruction: " << I << '\n';
				    MemoryAccessInfo info(MemoryAccessInfo::DirectStore, &I);
				    // TODO: Process the store instruction with the access info
				  }
				}
			}

			return false;
		}
  };
}

char LoadStoreAnalysisPass::ID = 0;

// Register the pass with the LLVM Pass Manager
static RegisterPass<LoadStoreAnalysisPass> X(
    "loadstoreanalysis", "Load/Store Analysis Pass",
    false /* Only looks at CFG */,
    false /* Analysis Pass */);

