// Test that llvm-bcanalyzer recognizes the stream type of a PCH file.

// RUN: mkdir -p %t-dir
// Copying files allow for read-only checkouts to run this test.
// RUN: cp %S/Inputs/pragma-once2-pch.h %t-dir
// RUN: cp %S/Inputs/pragma-once2.h %t-dir
// RUN: %clang_cc1 -x c++-header -emit-pch -fmodule-format=raw -o %t %t-dir/pragma-once2-pch.h
// RUN: llvm-bcanalyzer -dump %t | FileCheck %s
//
// CHECK: Stream type: Clang Serialized AST
