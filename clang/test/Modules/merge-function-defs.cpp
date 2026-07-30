// RUN: rm -rf %t
// RUN: %clang_cc1 -I%S/Inputs/merge-function-defs -fmodules -fmodule-map-file=%S/Inputs/merge-function-defs/map -fmodules-cache-path=%t %s -emit-llvm-only

#include "b.h"

struct X {
  virtual void f();
};
inline void X::f() {}

X x;
