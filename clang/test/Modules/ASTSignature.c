// RUN: rm -rf %t
// RUN: %clang_cc1 -iquote %S/Inputs/ASTHash/ -fsyntax-only -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-strict-context-hash \
// RUN:   -fmodules-cache-path=%t -fdisable-module-hash %s
// RUN: cp %t/MyHeader2.pcm %t1.pcm
// RUN: rm -rf %t
// RUN: %clang_cc1 -iquote "/dev/null" -iquote %S/Inputs/ASTHash/ -fsyntax-only \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-strict-context-hash \
// RUN:   -fmodules-cache-path=%t -fdisable-module-hash %s
// RUN: cp %t/MyHeader2.pcm %t2.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t1.pcm > %t1.dump
// RUN: llvm-bcanalyzer --dump --disable-histogram %t2.pcm > %t2.dump
// RUN: cat %t1.dump %t2.dump | FileCheck %s

#include "my_header_2.h"

my_int var = 42;

// CHECK: [[AST_BLOCK_HASH:<AST_BLOCK_HASH .*>]]
// CHECK: [[SIGNATURE:<SIGNATURE .*>]]
// CHECK: [[AST_BLOCK_HASH]]
// CHECK-NOT: [[SIGNATURE]]
// The modules built by this test are designed to yield the same AST. If this
// test fails, it means that the AST block is has become non-relocatable.
