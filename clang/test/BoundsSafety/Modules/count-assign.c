// RUN: rm -rf %t
// RUN: %clang_cc1 -fbounds-safety -fmodules -fno-implicit-modules -x c -I%S/Inputs/count-assign -emit-module %S/Inputs/count-assign/module.modulemap -fmodule-name=ca -o %t/count-assign.pcm
// RUN: %clang_cc1 -fbounds-safety  -fmodules -fno-implicit-modules -x c -I%S/Inputs/count-assign -ast-dump-all -o - %s -fmodule-file=%t/count-assign.pcm | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety  -fmodules -fno-implicit-modules -x c -I%S/Inputs/count-assign -emit-llvm -o - %s -fmodule-file=%t/count-assign.pcm | FileCheck %s -check-prefix=LLVM
#include "count-assign-module.h"

int bar(void) {
    int arr[10];
    return foo(arr, 10);
}

// CHECK: |-FunctionDecl {{.*}} imported in ca used foo 'int (int *__single __counted_by(len), int)'
// CHECK: | |-ParmVarDecl [[PARM_DECL_PTR:0x[a-z0-9]*]] {{.*}} imported in ca used ptr 'int *__single __counted_by(len)':'int *__single'
// CHECK: | |-ParmVarDecl {{.*}} imported in ca used len 'int'
// CHECK: | | `-DependerDeclsAttr {{.*}} Implicit [[PARM_DECL_PTR]] 0

// CHECK: |-FunctionDecl {{.*}} imported in ca baz 'void ()'
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl {{.*}} imported in ca used len 'int'
// CHECK: |   |   `-DependerDeclsAttr {{.*}} Implicit [[LOCAL_PTR_REF:0x[a-z0-9]*]] 0
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[LOCAL_PTR_REF]] {{.*}} imported in ca ptr 'void *__single __sized_by(len)':'void *__single'

// Check that we do not accidentally set nullcheck on expr loaded from pcm.
// LLVM-NOT: boundscheck.null
