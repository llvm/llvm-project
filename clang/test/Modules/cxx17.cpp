// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++1z -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/cxx17 %s -verify -fno-modules-error-recovery

// expected-no-diagnostics
struct MergeExceptionSpec {
  ~MergeExceptionSpec();
} mergeExceptionSpec; // trigger evaluation of exception spec

#include "decls.h"

MergeExceptionSpec mergeExceptionSpec2;
