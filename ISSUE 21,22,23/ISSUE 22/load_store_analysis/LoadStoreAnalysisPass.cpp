#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>
#include <queue>

using namespace llvm;

namespace {
  // Define the pass class
  struct LoadStoreAnalysisPass : public FunctionPass {
    static char ID;
    
    LoadStoreAnalysisPass() : FunctionPass(ID) {}

    // Run the pass on each function
    bool runOnFunction(Function &F) override {
      errs() << "Function: " << F.getName() << '\n';

      // Map to track memory access info for each memory location
      std::unordered_map<Value*, MemoryAccessInfo> memoryAccessMap;

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (I.getOpcode() == Instruction::Load) {
            errs() << "Found a load instruction: " << I << '\n';
            processLoadInstruction(&I, memoryAccessMap);
          } else if (I.getOpcode() == Instruction::Store) {
            errs() << "Found a store instruction: " << I << '\n';
            processStoreInstruction(&I, memoryAccessMap);
          }
        }
      }

      // Report the results after analyzing the function
      reportResults(memoryAccessMap);

      return false;
    }

  private:
    // Data structure to track memory access info
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
      Type *memoryType;
      uint64_t memorySize;
      uint64_t totalMemoryAccessed;
      uint64_t memoryAccessFrequency;

      MemoryAccessInfo() = default; // Default constructor

      MemoryAccessInfo(AccessPattern pattern, Instruction *inst)
          : pattern(pattern), inst(inst),
            memoryType(nullptr), memorySize(0),
            totalMemoryAccessed(0), memoryAccessFrequency(0) {}

      MemoryAccessInfo(AccessPattern pattern, Instruction *inst, Type *type, uint64_t size)
          : pattern(pattern), inst(inst),
            memoryType(type), memorySize(size),
            totalMemoryAccessed(0), memoryAccessFrequency(0) {}
    };

    // Helper function to process load instructions
    void processLoadInstruction(Instruction *I, std::unordered_map<Value*, MemoryAccessInfo> &memoryAccessMap) {
      Value *pointerOperand = I->getOperand(0)->stripPointerCasts();
      if (pointerOperand->getType()->isPointerTy()) {
        MemoryAccessInfo info(getAccessPattern(I), I);
        Type *memoryType = I->getType();
        uint64_t memorySize = getMemorySize(memoryType);
        // Update memory details for this load instruction
        info.memoryType = memoryType;
        info.memorySize = memorySize;
        info.totalMemoryAccessed = memorySize;
        info.memoryAccessFrequency = 1;

        // Update memory access info for this memory location in the map
        updateMemoryAccessInfo(memoryAccessMap, pointerOperand, info);
      }
    }

    // Helper function to process store instructions
    void processStoreInstruction(Instruction *I, std::unordered_map<Value*, MemoryAccessInfo> &memoryAccessMap) {
      Value *pointerOperand = I->getOperand(1)->stripPointerCasts();
      if (pointerOperand->getType()->isPointerTy()) {
        MemoryAccessInfo info(getAccessPattern(I), I);
        Type *memoryType = I->getOperand(0)->getType();
        uint64_t memorySize = getMemorySize(memoryType);
        // Update memory details for this store instruction
        info.memoryType = memoryType;
        info.memorySize = memorySize;
        info.totalMemoryAccessed = memorySize;
        info.memoryAccessFrequency = 1;

        // Update memory access info for this memory location in the map
        updateMemoryAccessInfo(memoryAccessMap, pointerOperand, info);
      }
    }

    // Helper function to update memory access info for a memory location in the map
    void updateMemoryAccessInfo(std::unordered_map<Value*, MemoryAccessInfo> &memoryAccessMap,
                                Value *pointerOperand, const MemoryAccessInfo &info) {
      if (memoryAccessMap.find(pointerOperand) == memoryAccessMap.end()) {
        memoryAccessMap[pointerOperand] = info;
      } else {
        MemoryAccessInfo &existingInfo = memoryAccessMap[pointerOperand];
        existingInfo.totalMemoryAccessed += info.totalMemoryAccessed;
        existingInfo.memoryAccessFrequency += info.memoryAccessFrequency;
      }
    }

    // Helper function to get the access pattern for an instruction
    MemoryAccessInfo::AccessPattern getAccessPattern(Instruction *I) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        return MemoryAccessInfo::PointerArithmetic;
      } else if (auto *load = dyn_cast<LoadInst>(I)) {
        return MemoryAccessInfo::DirectLoad;
      } else if (auto *store = dyn_cast<StoreInst>(I)) {
        return MemoryAccessInfo::DirectStore;
      }
      return MemoryAccessInfo::Unknown;
    }

    // Helper function to calculate memory size for a type
    uint64_t getMemorySize(Type *type) {
      if (auto *pointerType = dyn_cast<PointerType>(type)) {
        type = pointerType->getElementType();
      }

      if (auto *arrayType = dyn_cast<ArrayType>(type)) {
        return arrayType->getNumElements() * getMemorySize(arrayType->getElementType());
      } else if (auto *structType = dyn_cast<StructType>(type)) {
        uint64_t size = 0;
        for (unsigned i = 0, e = structType->getNumElements(); i != e; ++i) {
          size += getMemorySize(structType->getElementType(i));
        }
        return size;
      } else {
        return type->getPrimitiveSizeInBits() / 8;
      }
    }

    // Helper function to report the results after analyzing the function
    void reportResults(std::unordered_map<Value*, MemoryAccessInfo> &memoryAccessMap) {
      errs() << "Memory Access Results:\n";
      for (const auto &pair : memoryAccessMap) {
        Value *memoryLocation = pair.first;
        const MemoryAccessInfo &accessInfo = pair.second;
        errs() << "Memory Location: " << *memoryLocation << '\n';
        errs() << "Memory Type: " << *accessInfo.memoryType << '\n';
        errs() << "Memory Size: " << accessInfo.memorySize << " bytes\n";
        errs() << "Total Memory Accessed: " << accessInfo.totalMemoryAccessed << " bytes\n";
        errs() << "Memory Access Frequency: " << accessInfo.memoryAccessFrequency << " times\n";
        errs() << "Access Pattern: " << accessInfo.pattern << '\n';
        errs() << "---------------------------\n";
      }
    }
  };
}

char LoadStoreAnalysisPass::ID = 0;

// Register the pass with the LLVM Pass Manager
static RegisterPass<LoadStoreAnalysisPass> X(
    "loadstoreanalysis", "Load/Store Analysis Pass",
    false /* Only looks at CFG */,
    false /* Analysis Pass */);

