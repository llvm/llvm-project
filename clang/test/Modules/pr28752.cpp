// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -nostdsysteminc -I%S/Inputs/PR28752 -verify %s
// RUN: %clang_cc1 -std=c++11 -nostdsysteminc -fmodules -fmodule-map-file=%S/Inputs/PR28752/Subdir1/module.modulemap -fmodule-map-file=%S/Inputs/PR28752/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR28752 -I%S/Inputs/PR28752/Subdir1 -verify %s

#include "a.h"
#include "Subdir1/c.h"
#include <vector>

class TClingClassInfo {
  std::vector<int> fIterStack;
};

TClingClassInfo *a;
class TClingBaseClassInfo {
  TClingBaseClassInfo() { new TClingClassInfo(*a); }
};

// expected-no-diagnostics

