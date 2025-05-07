

// RUN: %clang_cc1 -fbounds-safety -ast-dump -verify %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -verify %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef char * cptr_t;
typedef long long int64_t;

void test_fdecl1(cptr_t __counted_by(len), int len);
// CHECK: FunctionDecl {{.*}} test_fdecl1 'void (char *__single __counted_by(len), int)'
// CHECK: |-ParmVarDecl {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: `-ParmVarDecl {{.*}} used len 'int'
// CHECK:   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0


void test_fdecl2(unsigned size, cptr_t __sized_by(size));
// CHECK: FunctionDecl {{.*}} test_fdecl2 'void (unsigned int, char *__single __sized_by(size))'
// CHECK: |-ParmVarDecl {{.*}} used size 'unsigned int'
// CHECK: | `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK: `-ParmVarDecl {{.*}} 'char *__single __sized_by(size)':'char *__single'

void test_fdecl_err1(int64_t __counted_by(len), int len);
// expected-error@-1{{'__counted_by' attribute only applies to pointer arguments}}

int glen;
void test_fdecl_err2(cptr_t __counted_by(glen));
// expected-error@-1{{count expression in function declaration may only reference parameters of that function}}

struct test_count_fields {
    cptr_t __sized_by(len) ptr;
    int64_t len;
};
// CHECK: RecordDecl {{.*}} struct test_count_fields definition
// CHECK: |-FieldDecl {{.*}} ptr 'char *__single __sized_by(len)':'char *__single'
// CHECK: `-FieldDecl {{.*}} referenced len 'int64_t':'long long'
// CHECK:   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0

struct test_range_fields {
    cptr_t __ended_by(end) iter;
    cptr_t __ended_by(iter) start;
    cptr_t end;
};
// CHECK: RecordDecl {{.*}} struct test_range_fields definition
// CHECK: FieldDecl {{.*}} referenced iter 'char *__single __ended_by(end) /* __started_by(start) */ ':'char *__single'
// CHECK: FieldDecl {{.*}} referenced start 'char *__single __ended_by(iter)':'char *__single'
// CHECK: FieldDecl {{.*}} referenced end 'char *__single /* __started_by(iter) */ ':'char *__single'
