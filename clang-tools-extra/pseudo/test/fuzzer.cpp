// RUN: clang-pseudo-fuzzer -grammar=%cxx-bnf-file -print %s | FileCheck %s
int x;
// CHECK: translation-unit := declaration-seq
// CHECK: builtin-type := INT
