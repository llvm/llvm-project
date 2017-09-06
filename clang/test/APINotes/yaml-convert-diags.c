// RUN: rm -rf %t
// RUN: not %clang_cc1 -fsyntax-only -fapinotes  %s -I %S/Inputs/BrokenHeaders2 2>&1 | FileCheck %s

#include "SomeBrokenLib.h"

// CHECK: error: multiple definitions of global function 'do_something_with_pointers'
