// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck --check-prefix=BS %s
// RUN: %clang_cc1 -fbounds-safety -fexperimental-bounds-safety-objc -x objective-c -ast-dump %s 2>&1 | FileCheck --check-prefix=BS %s
// RUN: %clang_cc1 -fbounds-safety -fexperimental-bounds-safety-cxx -x c++ -ast-dump %s 2>&1 | FileCheck --check-prefix=BS %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck --check-prefix=BSA %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck --check-prefix=BSA %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck --check-prefix=BSA %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck --check-prefix=BSA %s

#include <ptrcheck.h>

// BS: VarDecl {{.+}} func_ptr_dd 'void (*__single)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// BSA: VarDecl {{.+}} func_ptr_dd 'void (*)(void * __ended_by(end), void * /* __started_by(start) */ )'
void (*func_ptr_dd)(void *__ended_by(end) start, void *end);

// BS: VarDecl {{.+}} func_ptr_di 'void (*__single)(void *__single __ended_by(*end), void *__single /* __started_by(start) */ *__single)'
// BSA: VarDecl {{.+}} func_ptr_di 'void (*)(void * __ended_by(*end), void * /* __started_by(start) */ *)'
void (*func_ptr_di)(void *__ended_by(*end) start, void **end);

// BS: VarDecl {{.+}} func_ptr_id 'void (*__single)(void *__single __ended_by(end)*__single, void *__single /* __started_by(*start) */ )'
// BSA: VarDecl {{.+}} func_ptr_id 'void (*)(void * __ended_by(end)*, void * /* __started_by(*start) */ )'
void (*func_ptr_id)(void *__ended_by(end) *start, void *end);

// BS: VarDecl {{.+}} func_ptr_ii 'void (*__single)(void *__single __ended_by(*end)*__single, void *__single /* __started_by(*start) */ *__single)'
// BSA: VarDecl {{.+}} func_ptr_ii 'void (*)(void * __ended_by(*end)*, void * /* __started_by(*start) */ *)'
void (*func_ptr_ii)(void *__ended_by(*end) *start, void **end);

void foo(void) {
  // BS: CStyleCastExpr {{.+}} 'void (*)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
  // BSA: CStyleCastExpr {{.+}} 'void (*)(void * __ended_by(end), void * /* __started_by(start) */ )'
  (void (*)(void *__ended_by(end) start, void *end))0;
}
