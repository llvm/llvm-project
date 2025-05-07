// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=return_size -ast-dump -verify %s > %t.c.ast_dump.txt 2>&1
// RUN: FileCheck --input-file=%t.c.ast_dump.txt %S/bounds-attributed-in-return-null-system-header.h
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fbounds-safety-bringup-missing-checks=return_size -ast-dump -verify %s > %t.objc.ast_dump.txt 2>&1
// RUN: FileCheck --input-file=%t.objc.ast_dump.txt %S/bounds-attributed-in-return-null-system-header.h

#include "bounds-attributed-in-return-null-system-header.h"

// expected-no-diagnostics

// AST CHECK lines are in the included header file
int* __counted_by(count) test_explicit_unspecified_cast_0(int count) {
  return inline_header_ret_explicit_unspecified_cast_0(count);
}

// AST CHECK lines are in the included header file
int* __counted_by(count) test_explicit_unsafe_indexable_cast_0(int count) {
  return inline_header_ret_explicit_unsafe_indexable_cast_0(count);
}

// AST CHECK lines are in the included header file
int* __counted_by(count) test_0(int count) {
  return inline_header_ret_0(count);
}

// AST CHECK lines are in the included header file
int* __counted_by(count) test_void_star_unspecified_0(int count) {
  return inline_header_ret_void_star_unspecified_0(count);
}

// AST CHECK lines are in the included header file
int* __counted_by(count) test_void_star_unsafe_indexable_0(int count) {
  return inline_header_ret_void_star_unsafe_indexable_0(count);
}
