// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: %clang_cc1 -std=c++20 %s -emit-module-interface -o %t/a.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/a.pcm > %t/a.dump
// RUN: cat %t/a.dump | FileCheck %s
//
// RUN: %clang_cc1 -std=c++20 %s -emit-reduced-module-interface -o %t/a.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/a.pcm > %t/a.dump
// RUN: cat %t/a.dump | FileCheck %s

export module a;
// Contain something at least to make sure the compiler won't
// optimize this out.
export int a = 43;

// CHECK:  <DECLTYPES_BLOCK
// CHECK-NOT: <DECL_TYPEDEF
// CHECK:    <TYPE_TYPEDEF
// CHECK:    <DECL_CONTEXT_LEXICAL
// CHECK:    <DECL_EXPORT
// CHECK:    <TYPE_RECORD
// CHECK:    <DECL_VAR
// CHECK:    <EXPR_INTEGER_LITERAL
// CHECK:    <STMT_STOP
// CHECK:  </DECLTYPES_BLOCK>
