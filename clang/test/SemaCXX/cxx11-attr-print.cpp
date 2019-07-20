// RUN: %clang_cc1 -std=c++11 -ast-print -fms-extensions %s | FileCheck %s
//
// CHECK: int x __attribute__((aligned(4)));
int x __attribute__((aligned(4)));

// FIXME: Print this at a valid location for a __declspec attr.
// CHECK: int y __declspec(align(4));
__declspec(align(4)) int y;

// CHECK: int z {{\[}}[gnu::aligned(4)]];
int z [[gnu::aligned(4)]];

// CHECK: __attribute__((deprecated("warning")));
int a __attribute__((deprecated("warning")));

// CHECK: int b {{\[}}[gnu::deprecated("warning")]];
int b [[gnu::deprecated("warning")]];

// CHECK: __declspec(deprecated("warning"))
__declspec(deprecated("warning")) int c;

// CHECK: int d {{\[}}[deprecated("warning")]];
int d [[deprecated("warning")]];

// CHECK: __attribute__((deprecated("warning", "fixit")));
int e __attribute__((deprecated("warning", "fixit")));

// CHECK: int cxx11_alignas alignas(4);
alignas(4) int cxx11_alignas;

// CHECK: int c11_alignas _Alignas(alignof(int));
_Alignas(int) int c11_alignas;

// CHECK: void foo() __attribute__((const));
void foo() __attribute__((const));

// CHECK: void bar() __attribute__((__const));
void bar() __attribute__((__const));

// FIXME: It's unfortunate that the string literal prints with the below three
// cases given that the string is only exposed via the [[nodiscard]] spelling.
// CHECK: int f1() __attribute__((warn_unused_result("")));
int f1() __attribute__((warn_unused_result));

// CHECK: {{\[}}[clang::warn_unused_result("")]];
int f2 [[clang::warn_unused_result]] ();

// CHECK: {{\[}}[gnu::warn_unused_result("")]];
int f3 [[gnu::warn_unused_result]] ();

// FIXME: ast-print need to print C++11
// attribute after function declare-id.
// CHECK: {{\[}}[noreturn]];
void f4 [[noreturn]] ();

// CHECK: __attribute__((gnu_inline));
inline void f6() __attribute__((gnu_inline));

// CHECK: {{\[}}[gnu::gnu_inline]];
inline void f7 [[gnu::gnu_inline]] ();

// arguments printing
// CHECK: __attribute__((format(printf, 2, 3)));
void f8 (void *, const char *, ...) __attribute__ ((format (printf, 2, 3)));

// CHECK: int m __attribute__((aligned(4
// CHECK: int n alignas(4
// CHECK: static int f() __attribute__((pure))
// CHECK: static int g() {{\[}}[gnu::pure]]
template <typename T> struct S {
  __attribute__((aligned(4))) int m;
  alignas(4) int n;
  __attribute__((pure)) static int f() {
    return 0;
  }
  [[gnu::pure]] static int g() {
    return 1;
  }
};

// CHECK: int m __attribute__((aligned(4
// CHECK: int n alignas(4
// CHECK: static int f() __attribute__((pure))
// CHECK: static int g() {{\[}}[gnu::pure]]
template struct S<int>;

// CHECK: using Small2 {{\[}}[gnu::mode(byte)]] = int;
using Small2 [[gnu::mode(byte)]] = int;
