// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

/* typeof */

// CHECK:      FunctionDecl {{.+}} to_cb_in_in 'void (int *{{.*}} __counted_by(len), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'int *{{.*}} __counted_by(len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
// CHECK-NEXT: FunctionDecl {{.+}} to_cb_in_in_x 'void (int *{{.*}} __counted_by(len), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void to_cb_in_in(int *__counted_by(len) p, int len);
__typeof__(to_cb_in_in) to_cb_in_in_x;

// CHECK:      FunctionDecl {{.+}} to_cb_in_out 'void (int *{{.*}} __counted_by(*len), int *{{.*}})'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'int *{{.*}} __counted_by(*len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used len 'int *{{.*}}'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 0
// CHECK-NEXT: FunctionDecl {{.+}} to_cb_in_out_x 'void (int *{{.*}} __counted_by(*len), int *{{.*}})'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(*len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int *{{.*}}'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 0
void to_cb_in_out(int *__counted_by(*len) p, int *len);
__typeof__(to_cb_in_out) to_cb_in_out_x;

// CHECK:      FunctionDecl {{.+}} to_cb_out_in 'void (int *{{.*}} __counted_by(len)*{{.*}}, int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'int *{{.*}} __counted_by(len)*{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 1
// CHECK-NEXT: FunctionDecl {{.+}} to_cb_out_in_x 'void (int *{{.*}} __counted_by(len)*{{.*}}, int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(len)*{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 1
void to_cb_out_in(int *__counted_by(len) * p, int len);
__typeof__(to_cb_out_in) to_cb_out_in_x;

// CHECK:      FunctionDecl {{.+}} to_cb_out_out 'void (int *{{.*}} __counted_by(*len)*{{.*}}, int *{{.*}})'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'int *{{.*}} __counted_by(*len)*{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used len 'int *{{.*}}'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 1
// CHECK-NEXT: FunctionDecl {{.+}} to_cb_out_out_x 'void (int *{{.*}} __counted_by(*len)*{{.*}}, int *{{.*}})'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(*len)*{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int *{{.*}}'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 1
void to_cb_out_out(int *__counted_by(*len) * p, int *len);
__typeof__(to_cb_out_out) to_cb_out_out_x;

// CHECK:      FunctionDecl {{.+}} to_cb_ret 'int *{{.*}} __counted_by(len)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used len 'int'
// CHECK-NEXT: FunctionDecl {{.+}} to_cb_ret_x 'int *{{.*}} __counted_by(len)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
int *__counted_by(len) to_cb_ret(int len);
__typeof__(to_cb_ret) to_cb_ret_x;

// CHECK:      FunctionDecl {{.+}} to_sb_in_in 'void (void *{{.*}} __sized_by(size), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'void *{{.*}} __sized_by(size)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used size 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
// CHECK-NEXT: FunctionDecl {{.+}} to_sb_in_in_x 'void (void *{{.*}} __sized_by(size), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'void *{{.*}} __sized_by(size)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used size 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void to_sb_in_in(void *__sized_by(size) p, int size);
__typeof__(to_sb_in_in) to_sb_in_in_x;

// CHECK:      FunctionDecl {{.+}} to_cbn_in_in 'void (int *{{.*}} __counted_by_or_null(len), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} p 'int *{{.*}} __counted_by_or_null(len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
// CHECK-NEXT: FunctionDecl {{.+}} to_cbn_in_in_x 'void (int *{{.*}} __counted_by_or_null(len), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by_or_null(len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void to_cbn_in_in(int *__counted_by_or_null(len) p, int len);
__typeof__(to_cbn_in_in) to_cbn_in_in_x;

// CHECK:      FunctionDecl {{.+}} to_eb 'void (int *{{.*}} __ended_by(end), int *{{.*}} /* __started_by(start) */ )'
// CHECK-NEXT: |-ParmVarDecl {{.+}} used start 'int *{{.*}} __ended_by(end)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used end 'int *{{.*}} /* __started_by(start) */ '
// CHECK-NEXT: FunctionDecl {{.+}} to_eb_x 'void (int *{{.*}} __ended_by(end), int *{{.*}} /* __started_by(start) */ )'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit used start 'int *{{.*}} __ended_by(end)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used end 'int *{{.*}} /* __started_by(start) */ '
void to_eb(int *__ended_by(end) start, int *end);
__typeof__(to_eb) to_eb_x;

/* typedef */

// CHECK:      TypedefDecl {{.+}} td_cb_in_in 'void (int *{{.*}} __counted_by(len), int)'
// CHECK:      FunctionDecl {{.+}} tb_cb_in_in_x 'void (int *{{.*}} __counted_by(len), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
typedef void(td_cb_in_in)(int *__counted_by(len) p, int len);
td_cb_in_in tb_cb_in_in_x;

// CHECK:      TypedefDecl {{.+}} td_cb_in_out 'void (int *{{.*}} __counted_by(*len), int *{{.*}})'
// CHECK:      FunctionDecl {{.+}} td_cb_in_out_x 'void (int *{{.*}} __counted_by(*len), int *{{.*}})'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(*len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int *{{.*}}'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 0
typedef void(td_cb_in_out)(int *__counted_by(*len) p, int *len);
td_cb_in_out td_cb_in_out_x;

// CHECK:      TypedefDecl {{.+}} td_cb_out_in 'void (int *{{.*}} __counted_by(len)*{{.*}}, int)'
// CHECK:      FunctionDecl {{.+}} td_cb_out_in_x 'void (int *{{.*}} __counted_by(len)*{{.*}}, int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(len)*{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 1
typedef void(td_cb_out_in)(int *__counted_by(len) * p, int len);
td_cb_out_in td_cb_out_in_x;

// CHECK:      TypedefDecl {{.+}} td_cb_out_out 'void (int *{{.*}} __counted_by(*len)*{{.*}}, int *{{.*}})'
// CHECK:      FunctionDecl {{.+}} td_cb_out_out_x 'void (int *{{.*}} __counted_by(*len)*{{.*}}, int *{{.*}})'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by(*len)*{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int *{{.*}}'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 1
typedef void(td_cb_out_out)(int *__counted_by(*len) * p, int *len);
td_cb_out_out td_cb_out_out_x;

// CHECK:      TypedefDecl {{.+}} td_cb_ret 'int *{{.*}} __counted_by(len)(int)'
// CHECK:      FunctionDecl {{.+}} td_cb_ret_x 'int *{{.*}} __counted_by(len)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
typedef int *__counted_by(len)(td_cb_ret)(int len);
td_cb_ret td_cb_ret_x;

// CHECK:      TypedefDecl {{.+}} td_sb_in_in 'void (void *{{.*}} __sized_by(size), int)'
// CHECK:      FunctionDecl {{.+}} tb_sb_in_in_x 'void (void *{{.*}} __sized_by(size), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'void *{{.*}} __sized_by(size)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used size 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
typedef void(td_sb_in_in)(void *__sized_by(size) p, int size);
td_sb_in_in tb_sb_in_in_x;

// CHECK:      TypedefDecl {{.+}} td_cbn_in_in 'void (int *{{.*}} __counted_by_or_null(len), int)'
// CHECK:      FunctionDecl {{.+}} tb_cbn_in_in_x 'void (int *{{.*}} __counted_by_or_null(len), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit 'int *{{.*}} __counted_by_or_null(len)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used len 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
typedef void(td_cbn_in_in)(int *__counted_by_or_null(len) p, int len);
td_cbn_in_in tb_cbn_in_in_x;

// CHECK:      TypedefDecl {{.+}} td_eb 'void (int *{{.*}} __ended_by(end), int *{{.*}} /* __started_by(start) */ )'
// CHECK:      FunctionDecl {{.+}} td_eb_x 'void (int *{{.*}} __ended_by(end), int *{{.*}} /* __started_by(start) */ )'
// CHECK-NEXT: |-ParmVarDecl {{.+}} implicit used start 'int *{{.*}} __ended_by(end)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} implicit used end 'int *{{.*}} /* __started_by(start) */ '
typedef void(td_eb)(int *__ended_by(end) start, int *end);
td_eb td_eb_x;
