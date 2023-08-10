#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

const int NUM_RANGES = 3;

struct FragileCluster {
  int start;
  int end;
} Ranges[NUM_RANGES];

// Hardcoded ranges for testing
int pr[][2] = {
  {9, 14},
  {2, 6},
  {1, 22}
};

namespace {

struct MyPass : public PassInfoMixin<MyPass> {

  // Helper function which checks if two ranges overlap. Ranges are (A1, A2) and (B1, B2).
  bool doLocationsOverlap(int A1, int A2, int B1, int B2) {
    if (A1<=B2 && B1<=A2)
        return true;
    return false;
  }

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    // Intializing ranges with hardcoded ranges. Required only for testing.
    for(int i=0; i<NUM_RANGES;i++){
      Ranges[i].start = pr[i][0];
      Ranges[i].end = pr[i][1];
    }
    // Line number of last line in function
    int maxLineInFun = 0;
    // Number of lines in each function
    int numLines;
    // Length of signature string of function
    int signature_len;
    // Length of string containing parameters of fucntion
    int parameters_len;

    // Iterating over basic blocks of function
    for(auto &BB: F){
      DebugLoc BBEndLoc = BB.back().getDebugLoc();
      int BBEndLine = BBEndLoc.getLine();
      if(BBEndLine>maxLineInFun)
        maxLineInFun = BBEndLine;
    }

    auto *subprogram = F.getSubprogram();
    if(!subprogram)
      errs()<<"Has no subprogram\n";
    
    FunctionType *ft = F.getFunctionType();

    // If a function doesn't return (i.e has void return type or is the main function) then the last line will point
    // to the closing bracket. Otherwise it will point to the return statement so we need to add 1 for the closing
    // bracket.
    if(!(F.getName()=="main" || ft->getReturnType()->isVoidTy()))
      maxLineInFun = maxLineInFun + 1;
    
    for(auto BBRange: Ranges){
      if(doLocationsOverlap(subprogram->getLine(), maxLineInFun, BBRange.start, BBRange.end)){
        // The function overlaps with the range so print formatted output
        outs()<<"----------------------------------------------------------------------------------------------------";
        outs()<<"--------------------------------------------------\n";
        outs()<<"Source File\t\t"<<F.getParent()->getSourceFileName()<<"\n";
        outs()<<"Signature\t\t";
        outs()<<F.getName()<<" ";
        outs()<<*ft<<"\n";
        outs()<<"Parameter names\t\t";
        if(F.args().empty())
          outs()<<"<None>";
        else {
          for (auto& A : F.args())
            outs()<<A.getName()<<" ";
        }
        outs()<<"\n";

        //Uncomment the commented lines in this block of code to print the attribute list of each function before and
        // after adding the attribute `isFragile`
        //outs()<<"Attribute list before adding isFragile\n";
        //F.getAttributes().print(outs());
        const char *s = "isFragile";
        const char *t = "true";
        F.addFnAttr(s, t);
        //outs()<<"Attribute list after adding isFragile\n";
        //F.getAttributes().print(outs());

        numLines = maxLineInFun - subprogram->getLine() + 1;
        outs() << "Number of lines\t\t"<<numLines<<"\n";
        break;
      }
    }
    return PreservedAnalyses::all();
  }
};

} // end anonymous namespace

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(createModuleToFunctionPassAdaptor(MyPass()));
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "fragile marker", "0.0.1", callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}