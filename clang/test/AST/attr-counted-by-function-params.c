// RUN: %clang_cc1 -verify %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -verify %s -ast-dump | FileCheck %s

#define __counted_by(f)  __attribute__((counted_by(f)))

int global_count;

// CHECK: FunctionDecl {{.*}} back_ref 'void (int, int * __counted_by(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'
void back_ref(int count, int *__counted_by(count) buf);

// A count written after the declarator, which late parsing parses at the end of
// the parameter list.
// CHECK: FunctionDecl {{.*}} trailing 'void (int, int * __counted_by(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'
void trailing(int count, int *buf __counted_by(count));

// A rejected count is not applied, and the function stays valid.
// CHECK: FunctionDecl {{.*}} not_a_param 'void (int *)'
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param(int *__counted_by(global_count) buf);

// A count on a K&R parameter declaration is ignored.
// CHECK: FunctionDecl {{.*}} knr 'void (int, int *)'
// expected-warning@+2{{a function definition without a prototype is deprecated}}
// expected-warning@+1{{'counted_by' attribute ignored}}
void knr(count, buf) int count; int *__counted_by(count) buf; {}
