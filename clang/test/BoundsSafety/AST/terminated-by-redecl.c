
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include "terminated-by-redecl.h"
// CHECK:|-FunctionDecl {{.*}} test_system_no_annot_argument 'void (int *)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'int *'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_nt_argument 'void (int *__single __terminated_by(0))'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_nt_argument_implicit_1 'void (const char *)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'const char *'
// CHECK-NEXT:|-FunctionDecl {{.*}}  test_system_nt_argument_implicit_2 'void (const char *)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'const char *'
// CHECK-NEXT:|-FunctionDecl {{.*}}  test_system_nt_argument_implicit_3 'void (const char *)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'const char *'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_no_annot_return 'int *()'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_nt_return 'int *__single __terminated_by(0)()'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_nt_return_implicit_1 'const char *()'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_nt_return_implicit_2 'const char *()'
// CHECK-NEXT:|-FunctionDecl {{.*}} test_system_nt_return_implicit_3 'const char *()'


const char *test();
const char *__null_terminated test();
// CHECK-NEXT:|-FunctionDecl {{.*}} test 'const char *__single __terminated_by(0)()'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test 'const char *__single __terminated_by(0)()'


void test_system_no_annot_argument(int *__null_terminated p);
void test_system_nt_argument(int *__null_terminated p);
void test_system_nt_argument_implicit_1(const char *p);
void test_system_nt_argument_implicit_2(const char *__single p);
void test_system_nt_argument_implicit_3(const char *__null_terminated p);
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_no_annot_argument 'void (int *__single __terminated_by(0))'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_argument 'void (int *__single __terminated_by(0))'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'int *__single __terminated_by(0)':'int *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_argument_implicit_1 'void (const char *__single __terminated_by(0))'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'const char *__single __terminated_by(0)':'const char *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_argument_implicit_2 'void (const char *__single)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'const char *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_argument_implicit_3 'void (const char *__single __terminated_by(0))'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} p 'const char *__single __terminated_by(0)':'const char *__single'


int *__null_terminated test_system_no_annot_return();
int *__null_terminated test_system_nt_return();
const char *test_system_nt_return_implicit_1();
const char *__unsafe_indexable test_system_nt_return_implicit_2();
const char *__null_terminated test_system_nt_return_implicit_3();

// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_no_annot_return 'int *__single __terminated_by(0)()'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_return 'int *__single __terminated_by(0)()'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_return_implicit_1 'const char *__single __terminated_by(0)()'
// CHECK-NEXT:|-FunctionDecl {{.*}} prev {{.*}} test_system_nt_return_implicit_2 'const char *__unsafe_indexable()'
// CHECK-NEXT:`-FunctionDecl {{.*}} prev {{.*}} test_system_nt_return_implicit_3 'const char *__single __terminated_by(0)()'
