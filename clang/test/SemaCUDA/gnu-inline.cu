// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// expected-no-diagnostics

// Check that we can handle gnu_inline functions when compiling in CUDA mode.

void foo();
inline __attribute__((gnu_inline)) void bar() { foo(); }
