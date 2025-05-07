

// RUN: %clang_cc1 -ast-dump -fbounds-safety -verify %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Foo {
    void *__sized_by(sizeof(int) * num_elements) vptr;
    unsigned int num_elements;

    unsigned long long number_of_bits;
    const unsigned char *__sized_by((number_of_bits + 7) & -7) buffer;

    char *__sized_by(sizeof(char) * 100) cptr;
};

// CHECK: RecordDecl {{.*}} struct Foo definition
// CHECK: |-FieldDecl [[VPTR_PTR:0x[0-9a-z]*]] {{.*}} vptr 'void *__single __sized_by(4UL * num_elements)':'void *__single'
// CHECK: |-FieldDecl {{.*}} referenced num_elements 'unsigned int'
// CHECK: | `-DependerDeclsAttr {{.*}} Implicit [[VPTR_PTR]] 0
// CHECK: |-FieldDecl {{.*}} referenced number_of_bits 'unsigned long long'
// CHECK: | `-DependerDeclsAttr {{.*}} Implicit [[BUFFER_PTR:0x[0-9a-z]*]] 0
// CHECK: |-FieldDecl [[BUFFER_PTR]] {{.*}} buffer 'const unsigned char *__single __sized_by((number_of_bits + 7) & -7)':'const unsigned char *__single'
// CHECK: `-FieldDecl {{.*}} cptr 'char *__single __sized_by(100UL)':'char *__single'

static int g_arr[] = { 0, 1, 2 }; // expected-note{{'g_arr' declared here}}
static char g_char[100];
const unsigned char g_cchar[] = "Hello world!"; // expected-note{{'g_cchar' declared here}}
// confirms that the compiler treats this as compile-time constant.
struct Foo g_foo = {g_arr, sizeof(g_arr)/sizeof(int), 3, g_cchar, g_char};
// expected-error@+1{{initializing 'g_foo_ovf.buffer' of type 'const unsigned char *__single __sized_by((number_of_bits + 7) & -7)' (aka 'const unsigned char *__single') and size value of 17 with array 'g_cchar' (which has 13 bytes) always fails}}
struct Foo g_foo_ovf = {g_arr, sizeof(g_arr)/sizeof(int), 10, g_cchar, g_char};

struct Bar {
    int num;
    int *__counted_by(*(&num) + *&(*&num) + (-num)) iptr;
};

const float cf = 1.0f;
struct Baz {
    float f;
    int *__counted_by((int)f) ptr1;
    // FIXME: "*&" pattern with '&' wrapped in C-style casts is not supported yet.
    int *__counted_by(*(int*)(&f)) ptr2; // expected-error{{invalid argument expression to bounds attribute}}
    int *__counted_by(*(int*)(&cf)) ptr3; // expected-error{{invalid argument expression to bounds attribute}}
};

struct Bar g_bar = {3, g_arr};
// expected-error@+1{{initializing 'g_bar_ovf.iptr' of type 'int *__single __counted_by(num + num + (-num))' (aka 'int *__single') and count value of 4 with array 'g_arr' (which has 3 elements) always fails}}
struct Bar g_bar_ovf = {4, g_arr};

void test() {
    int arr[100];
    int *var;
    void *__sized_by(sizeof(var) * 10) vptr = arr;

    const int num = 10;
    int *__counted_by(*(&num) + *&(*&num)) iptr = arr;
    int *__counted_by((int)10.0f) iptr2 = arr;
    int *__counted_by(10.0f) iptr3 = arr; // expected-error{{'__counted_by' attribute requires an integer type argument}}

    iptr = iptr2; // this is not an error since 'num' is constant.
}

// CHECK: DeclStmt
// CHECK: `-VarDecl {{.*}} referenced var 'int *__bidi_indexable'
// CHECK: DeclStmt
// CHECK: `-VarDecl {{.*}} vptr 'void *__single __sized_by(240UL)':'void *__single'
// CHECK: DeclStmt
// CHECK: `-VarDecl {{.*}} used num 'const int' cinit
// CHECK:   `-IntegerLiteral {{.*}} 'int' 10
// CHECK: DeclStmt
// CHECK: `-VarDecl {{.*}} iptr 'int *__single __counted_by(20)':'int *__single'
// CHECK: DeclStmt
// CHECK: `-VarDecl {{.*}} iptr2 'int *__single __counted_by(10)':'int *__single'
// CHECK: DeclStmt
// CHECK: `-VarDecl {{.*}} iptr3 'int *__single __counted_by(0)':'int *__single'
