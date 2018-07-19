// See overflow.c for a description.

// REQUIRES: aarch64-target-arch
// RUN: %clang_scs %S/overflow.c -o %t -DITERATIONS=12
// RUN: %run %t | FileCheck %S/overflow.c
