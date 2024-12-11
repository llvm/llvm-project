// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>


// Decls in the system header don't have any attributes.

#include "system-header-merge-bounds-attributed.h"

// CHECK: FunctionDecl {{.+}} cb_in
// CHECK: |-ParmVarDecl {{.+}} cb_in_p 'int *'
// CHECK: `-ParmVarDecl {{.+}} len 'int'
// CHECK: FunctionDecl {{.+}} cb_out
// CHECK: |-ParmVarDecl {{.+}} cb_out_p 'int **'
// CHECK: `-ParmVarDecl {{.+}} len 'int'
// CHECK: FunctionDecl {{.+}} cb_out_count
// CHECK: |-ParmVarDecl {{.+}} cb_out_len_p 'int *'
// CHECK: `-ParmVarDecl {{.+}} len 'int *'
// CHECK: FunctionDecl {{.+}} cbn
// CHECK: |-ParmVarDecl {{.+}} cbn_p 'int *'
// CHECK: `-ParmVarDecl {{.+}} len 'int'
// CHECK: FunctionDecl {{.+}} sb
// CHECK: |-ParmVarDecl {{.+}} sb_p 'void *'
// CHECK: `-ParmVarDecl {{.+}} size 'int'
// CHECK: FunctionDecl {{.+}} eb
// CHECK: |-ParmVarDecl {{.+}} eb_p 'void *'
// CHECK: `-ParmVarDecl {{.+}} end 'void *'


// Check if we can override them.

void cb_in(int *__counted_by(len) cb_in_p, int len);
void cb_out(int *__counted_by(len) *cb_out_p, int len);
void cb_out_count(int *__counted_by(*len) cb_out_len_p, int *len);
void cbn(int *__counted_by_or_null(len) cbn_p, int len);
void sb(void *__sized_by(size) sb_p, int size);
void eb(void *__ended_by(end) eb_p, void *end);

// CHECK: FunctionDecl {{.+}} prev {{.+}} cb_in
// CHECK: |-ParmVarDecl {{.+}} cb_in_p 'int *{{.*}} __counted_by(len)'
// CHECK: `-ParmVarDecl {{.+}} used len 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} cb_out
// CHECK: |-ParmVarDecl {{.+}} cb_out_p 'int *{{.*}} __counted_by(len)*{{.*}}'
// CHECK: `-ParmVarDecl {{.+}} used len 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} cb_out_count
// CHECK: |-ParmVarDecl {{.+}} cb_out_len_p 'int *{{.*}} __counted_by(*len)'
// CHECK: `-ParmVarDecl {{.+}} used len 'int *{{.*}}'
// CHECK: FunctionDecl {{.+}} prev {{.+}} cbn
// CHECK: |-ParmVarDecl {{.+}} cbn_p 'int *{{.*}} __counted_by_or_null(len)'
// CHECK: `-ParmVarDecl {{.+}} used len 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} sb
// CHECK: |-ParmVarDecl {{.+}} sb_p 'void *{{.*}} __sized_by(size)'
// CHECK: `-ParmVarDecl {{.+}} used size 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} eb
// CHECK: |-ParmVarDecl {{.+}} used eb_p 'void *{{.*}} __ended_by(end)'
// CHECK: `-ParmVarDecl {{.+}} used end 'void *{{.*}} /* __started_by(eb_p) */ '


// Check if the attributes are merged.

#include "system-header-merge-bounds-attributed.h"

// CHECK: FunctionDecl {{.+}} prev {{.+}} cb_in
// CHECK: |-ParmVarDecl {{.+}} cb_in_p 'int *{{.*}} __counted_by(len)'
// CHECK: `-ParmVarDecl {{.+}} used len 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} cb_out
// CHECK: |-ParmVarDecl {{.+}} cb_out_p 'int *{{.*}} __counted_by(len)*{{.*}}'
// CHECK: `-ParmVarDecl {{.+}} used len 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} cb_out_count
// CHECK: |-ParmVarDecl {{.+}} cb_out_len_p 'int *{{.*}} __counted_by(*len)'
// CHECK: `-ParmVarDecl {{.+}} used len 'int *{{.*}}'
// CHECK: FunctionDecl {{.+}} prev {{.+}} cbn
// CHECK: |-ParmVarDecl {{.+}} cbn_p 'int *{{.*}} __counted_by_or_null(len)'
// CHECK: `-ParmVarDecl {{.+}} used len 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} sb
// CHECK: |-ParmVarDecl {{.+}} sb_p 'void *{{.*}} __sized_by(size)'
// CHECK: `-ParmVarDecl {{.+}} used size 'int'
// CHECK: FunctionDecl {{.+}} prev {{.+}} eb
// CHECK: |-ParmVarDecl {{.+}} used eb_p 'void *{{.*}} __ended_by(end)'
// CHECK: `-ParmVarDecl {{.+}} used end 'void *{{.*}} /* __started_by(eb_p) */ '
