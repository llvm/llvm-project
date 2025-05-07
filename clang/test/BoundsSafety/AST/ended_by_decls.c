// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fexperimental-bounds-safety-attributes -x c %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fexperimental-bounds-safety-attributes -x c++ %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fexperimental-bounds-safety-attributes -x objective-c %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fexperimental-bounds-safety-attributes -x objective-c++ %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: RecordDecl {{.*}} struct S definition
// CHECK: |-FieldDecl {{.*}} referenced end 'int *{{.*}}/* __started_by(iter) */ '
// CHECK: |-FieldDecl {{.*}} referenced start 'int *{{.*}}__ended_by(iter)'
// CHECK: `-FieldDecl {{.*}} referenced iter 'int *{{.*}}__ended_by(end) /* __started_by(start) */ '
struct S {
    int *end;
    int *__ended_by(iter) start;
    int *__ended_by(end) iter;
};

// CHECK: FunctionDecl {{.*}} foo 'void (int *{{.*}}__ended_by(end), int *{{.*}}/* __started_by(start) */ )'
// CHECK: |-ParmVarDecl {{.*}} used start 'int *{{.*}}__ended_by(end)'
// CHECK: `-ParmVarDecl {{.*}} used end 'int *{{.*}}/* __started_by(start) */ '
void foo(int *__ended_by(end) start, int* end);

// CHECK: FunctionDecl {{.*}} foo_cptr_end 'void (int *{{.*}}__ended_by(end), char *{{.*}}/* __started_by(start) */ )'
// CHECK: |-ParmVarDecl {{.*}} used start 'int *{{.*}}__ended_by(end)'
// CHECK: `-ParmVarDecl {{.*}} used end 'char *{{.*}}/* __started_by(start) */ '
void foo_cptr_end(int *__ended_by(end) start, char* end);

// CHECK: FunctionDecl {{.*}} foo_seq 'void (int *{{.*}}__ended_by(next), int *{{.*}}__ended_by(end) /* __started_by(start) */ , char *{{.*}}/* __started_by(next) */ )'
// CHECK: |-ParmVarDecl {{.*}} used start 'int *{{.*}}__ended_by(next)'
// CHECK: |-ParmVarDecl {{.*}} used next 'int *{{.*}}__ended_by(end) /* __started_by(start) */ '
// CHECK: `-ParmVarDecl {{.*}} used end 'char *{{.*}}/* __started_by(next) */ '
void foo_seq(int *__ended_by(next) start, int *__ended_by(end) next, char* end);

// CHECK: FunctionDecl {{.*}} foo_out_start_out_end 'void (int *{{.*}}__ended_by(*out_end)*{{.*}}, int *{{.*}}/* __started_by(*out_start) */ *{{.*}})'
// CHECK: |-ParmVarDecl {{.*}} used out_start 'int *{{.*}}__ended_by(*out_end)*{{.*}}'
// CHECK: `-ParmVarDecl {{.*}} used out_end 'int *{{.*}}/* __started_by(*out_start) */ *{{.*}}'
void foo_out_start_out_end(int *__ended_by(*out_end) *out_start, int **out_end);

// CHECK: FunctionDecl {{.*}} foo_out_end 'void (int *{{.*}}__ended_by(*out_end), int *{{.*}}/* __started_by(start) */ *{{.*}})'
// CHECK: |-ParmVarDecl {{.*}} used start 'int *{{.*}}__ended_by(*out_end)':'int *{{.*}}'
// CHECK: `-ParmVarDecl {{.*}} used out_end 'int *{{.*}}/* __started_by(start) */ *{{.*}}'
void foo_out_end(int *__ended_by(*out_end) start, int **out_end);

// CHECK: FunctionDecl {{.*}} foo_ret_end 'int *{{.*}}__ended_by(end)(int *{{.*}})'
// CHECK: `-ParmVarDecl {{.*}} used end 'int *{{.*}}'
int *__ended_by(end) foo_ret_end(int *end);

// CHECK: FunctionDecl {{.*}} foo_local_ended_by 'void ({{.*}})'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl {{.*}} used end 'int *{{.*}}/* __started_by(start) */ '
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl {{.*}} used start 'int *{{.*}}__ended_by(end)'
void foo_local_ended_by(void) {
  int *end;
  int *__ended_by(end) start;
}
