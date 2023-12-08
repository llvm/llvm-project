// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -I%S/Inputs/merge-typedefs -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-typedefs -verify %s

#include "b2.h"
#include "a1.h"

// expected-no-diagnostics
llvm::MachineDomTreeNode *p;
foo2_t f2t;
