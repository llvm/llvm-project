// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR28794 -verify %s
// RUN: %clang_cc1 -std=c++11 -fmodules -fmodule-map-file=%S/Inputs/PR28794/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR28794/ -verify %s

#include "Subdir/Empty.h"
#include "LibAHeader.h"

BumpPtrAllocatorImpl<> &getPreprocessorAllocator();
class B {
  struct ModuleMacroInfo {
    ModuleMacroInfo *getModuleInfo() {
      return new (getPreprocessorAllocator()) ModuleMacroInfo();
    }
  };
};

// expected-no-diagnostics
