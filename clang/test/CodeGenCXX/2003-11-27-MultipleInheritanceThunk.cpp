// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o -
// RUN: %clang_cc1 -triple %ms_abi_triple -fno-rtti -emit-llvm %s -o -


struct CallSite {
  int X;

  CallSite(const CallSite &CS);
};

struct AliasAnalysis {
  int TD;

  virtual int getModRefInfo(CallSite CS);
};


struct Pass {
  int X;
  virtual int foo();
};

struct AliasAnalysisCounter : public Pass, public AliasAnalysis {
  int getModRefInfo(CallSite CS) {
    return 0;
  }
};

AliasAnalysisCounter AAC;
