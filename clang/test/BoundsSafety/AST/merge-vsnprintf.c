// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -x c++ -fexperimental-bounds-safety-cxx -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stdarg.h>
#include <stddef.h>

#define __printf(string_index, first_to_check)                                 \
  __attribute__((__format__(__printf__, string_index, first_to_check)))

#ifndef __cplusplus
#define __restrict restrict
#endif

// CHECK: ParmVarDecl {{.+}} foo_args 'va_list':'char *'
int vsnprintf(char *__restrict __counted_by(__size) __str, size_t __size,
              const char *__restrict __format, va_list foo_args) __printf(3, 0);
