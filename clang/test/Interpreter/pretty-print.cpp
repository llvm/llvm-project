// RUN: clang-repl "int i = 10;" 'extern "C" int printf(const char*,...);' \
// RUN:            'auto r1 = printf("i = %d\n", i);' | FileCheck --check-prefix=CHECK-DRIVER %s
// The test is flaky with asan https://github.com/llvm/llvm-project/pull/148701.
// UNSUPPORTED: asan
// CHECK-DRIVER: i = 10
// RUN: cat %s | clang-repl -Xcc -std=c++11 -Xcc -fno-delayed-template-parsing | FileCheck %s
extern "C" int printf(const char*,...);

"ab"
// CHECK: (const char[3]) "ab"

123456
// CHECK-NEXT: (int) 123456

char ch[2] = {'1','a'}; ch
// CHECK-NEXT: (char[2]) { '1', 'a' }

char chnull[3] = {'1','a', '\0'}; chnull
// CHECK-NEXT: (char[3]) "1a"

char ch_arr[2][3][1] = {{{'a'}, {'b'}, {'c'}}, {{'d'}, {'e'}, {'f'}}}; ch_arr
// CHECK: (char[2][3][1]) { { { 'a' }, { 'b' }, { 'c' } }, { { 'd' }, { 'e' }, { 'f' } } }
struct S3 { int* p; S3() { p = new int(42); } ~S3() { delete p; } };
S3{}
// CHECK-NEXT: (S3) @0x{{[0-9a-f]+}}
S3 s3;
s3
// CHECK-NEXT: (S3 &) @0x{{[0-9a-f]+}}

struct S4 { ~S4() { printf("~S4()\n"); }};
S4{}
// CHECK-NEXT: (S4) @0x{{[0-9a-f]+}}
// TODO-CHECK-NEXT: ~S4()

enum Enum{ e1 = -12, e2, e3=33, e4, e5 = 33};
e2
// CHECK-NEXT: (Enum) (e2) : int -11
::e1
// CHECK-NEXT: (Enum) (e1) : int -12

enum class Color { R = 0, G, B };
Color::R
// CHECK-NEXT: (Color) (Color::R) : int 0


// Lambdas.

auto Lambda1 = []{};
Lambda1
// CHECK-NEXT: ((lambda) &) @0x{{[0-9a-f]+}}
[]{}
// CHECK-NEXT: ((lambda at input_line_{{[0-9]+}}:1:1)) @0x{{[0-9a-f]+}}

template<int n> struct F{ enum {RET=F<n-1>::RET*n} ; };
template<> struct F<0> { enum {RET = 1}; };
F<7>::RET
// CHECK-NEXT: (F<7>::(unnamed enum at input_line_{{[0-9]+}}:1:27)) (F<7>::RET) : unsigned int 5040

struct S5 { int foo() { return 42; }};
&S5::foo
// CHECK-NEXT: (int (S5::*)()) Function @0x{{[0-9a-f]+}}

// int i = 12;
// int &iref = i;
// iref
// // TODO-CHECK-NEXT: (int &) 12

// int &&rref = 100;
// rref

// // TODO-CHECK-NEXT: (int &&) 100

%quit

