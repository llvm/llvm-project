

// RUN: %clang_cc1 -fsyntax-only -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>
#include "system-header-terminated-by.h"

// CHECK:      FunctionDecl {{.+}} foo 'void (int *__single __terminated_by(0))'
// CHECK-NEXT: `-ParmVarDecl {{.+}} 'int *__single __terminated_by(0)':'int *__single'
