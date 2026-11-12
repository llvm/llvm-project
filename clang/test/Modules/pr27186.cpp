// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -I%S/Inputs/PR27186  -I%S/Inputs/PR27186/subdir/ -verify %s
// RUN: %clang_cc1  -nostdsysteminc -std=c++11 -fmodules  -fmodule-map-file=%S/Inputs/PR27186/module.modulemap -fmodules-cache-path=%t -I%S/Inputs/PR27186/ -verify %s

#include "Rtypes.h"

// expected-no-diagnostics
