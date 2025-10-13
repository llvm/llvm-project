// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -xc  | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -std=c++11 | FileCheck %s

// UNSUPPORTED: hwasan, msan


char c = 'a'; c
// CHECK: (char) 'a'

const char* c_str = "Hello, world!"; c_str
// CHECK-NEXT: (const char *) "Hello, world!"

c_str = "Goodbye, world!"; c_str
// CHECK-NEXT: (const char *) "Goodbye, world!"

const char* c_null_str = 0; c_null_str
// CHECK-NEXT: (const char *) 0

"Hello, world"
// CHECK-NEXT: ({{(const )?}}char[13]) "Hello, world"

int x = 42; x
// CHECK-NEXT: (int) 42

&x
// CHECK-NEXT: (int *) 0x{{[0-9a-f]+}}

x - 2
// CHECK-NEXT: (int) 40

float f = 4.2f; f
// CHECK-NEXT: (float) 4.20000f

double d = 4.21; d
// CHECK-NEXT: (double) 4.2100000

long double tau = 6.2831853; tau
// CHECK-NEXT: (long double) 6.28318530000L

int foo() { return 42; } foo()
// CHECK-NEXT: (int) 42

void bar(int a, float b) {} bar
// CHECK-NEXT: (void (int, float)) Function @0x{{[0-9a-f]+}}
// CHECK-NEXT: void bar(int a, float b) {

bar
// CHECK: (void (int, float)) Function @0x{{[0-9a-f]+}}
// CHECK-NEXT: void bar(int a, float b) {

// Arrays.

int arr[3] = {1,2,3}; arr
// CHECK: (int[3]) { 1, 2, 3 }

double darr[3][4] = { {1,2,3,4}, {5,6,7,8}, {9,10,11,12} }; darr
// CHECK-NEXT: (double[3][4]) { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } }

float farr[2][1] = { {0}, {3.14}}; farr
// CHECK-NEXT: (float[2][1]) { { 0.0f }, { 3.14000f } }

0./0.
// CHECK-NEXT: (double) nan

1.0f / 0.0f
// CHECK-NEXT: (float) inf

0.00001f
// CHECK-NEXT: (float) 1.00000e-05f

int * ptr = (int*)0x123; ptr
// CHECK-NEXT: (int *) 0x123

int * null_ptr = (int*)0; null_ptr
// CHECK-NEXT: (int *) 0x0

// TODO: _Bool, _Complex, _Atomic, and _BitInt
// union U { int I; float F; } u; u.I = 12; u.I
// TODO-CHECK-NEXT: (int) 12
// struct S1{} s1; s1
// TODO-CHECK-NEXT: (S1 &) @0x{{[0-9a-f]+}}

// struct S2 {int d;} E = {22}; E
// TODO-CHECK-NEXT: (struct S2 &) @0x{{[0-9a-f]+}}
// E.d
// TODO-CHECK-NEXT: (int) 22

%quit
