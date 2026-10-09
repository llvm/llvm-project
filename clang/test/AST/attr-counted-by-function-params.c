// RUN: %clang_cc1 -verify %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -verify %s -ast-dump | FileCheck %s

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

int global_count;

// CHECK: FunctionDecl {{.*}} back_ref 'void (int, int * __counted_by(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} buf 'int * __counted_by(count)':'int *'
void back_ref(int count, int *__counted_by(count) buf);

// An array parameter's count moves to the pointer it adjusts to.
// CHECK: FunctionDecl {{.*}} array 'void (int, int * __sized_by_or_null(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} arr 'int * __sized_by_or_null(count)':'int *'
void array(int count, int arr[] __sized_by_or_null(count));

// The array's qualifiers apply to its element (C99 6.7.3p8), also when they are
// outside the count, as when it is written in the declaration specifiers.
typedef int int_array[];
// CHECK: FunctionDecl {{.*}} array_qualified 'void (int, const int * __counted_by(count))'
// CHECK-NEXT: ParmVarDecl {{.*}} count 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} arr 'const int * __counted_by(count)':'const int *'
void array_qualified(int count, const int_array __counted_by(count) arr);

// A type attribute written after the count does not hide it.
// CHECK: FunctionDecl {{.*}} array_attr_after 'void (int, int * __counted_by(count))'
void array_attr_after(int count,
                      int arr[] __counted_by(count) __attribute__((btf_type_tag("t"))));
// Nor does one that comes from a macro.
#define __noderef __attribute__((noderef))
// CHECK: FunctionDecl {{.*}} array_macro_attr_after 'void (int, int * __counted_by(count))'
void array_macro_attr_after(int count, int arr[] __counted_by(count) __noderef);

// A count that comes with the array's type through '__typeof__' names a field,
// so it does not move to the pointer the parameter adjusts to.
struct fam { int n; int arr[] __counted_by(n); };
// CHECK: FunctionDecl {{.*}} typeof_fam 'void (int *)'
void typeof_fam(__typeof__(((struct fam *)0)->arr) arr);

// A second count is rejected, so the first stays on the pointer.
// CHECK: FunctionDecl {{.*}} repeated_array 'void (int, int, int * __counted_by(n))'
// expected-error@+1{{pointer cannot have more than one count attribute}}
void repeated_array(int n, int m, int arr[] __counted_by(n) __counted_by(m));

// An out parameter's count, here a dereferenced parameter, is on the pointer
// that the parameter points to.
// CHECK: FunctionDecl {{.*}} out 'void (int *, int * __counted_by(*len)*)'
void out(int *len, int *__counted_by(*len) *buf);

// A rejected count is not applied, and the function stays valid.
// CHECK: FunctionDecl {{.*}} not_a_param 'void (int *)'
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
void not_a_param(int *__counted_by(global_count) buf);

// A nested count is dropped; the parameter's own count stays.
// CHECK: FunctionDecl {{.*}} both_counted 'void (int, int, int ** __counted_by(m))'
// expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
void both_counted(int n, int m, int *__counted_by(n) *__counted_by(m) buf);

// A count on a K&R parameter declaration is dropped.
// CHECK: FunctionDecl {{.*}} knr 'void (int, int *)'
// expected-warning@+1{{a function definition without a prototype is deprecated}}
void knr(count, buf) int count; int *__counted_by(count) buf; {}
