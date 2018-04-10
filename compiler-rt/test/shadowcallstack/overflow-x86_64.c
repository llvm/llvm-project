// See overflow.c for a description.

// REQUIRES: x86_64-target-arch
// RUN: %clang_scs %S/overflow.c -o %t -DITERATIONS=12
// RUN: not --crash %run %t
