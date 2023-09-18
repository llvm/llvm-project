// RUN: %clang_cc1 -fms-extensions -std=c++11 -E %s -fuse-line-directives | FileCheck %s

// Test that we properly expand the C99 _Pragma and Microsoft __pragma
// into #pragma directives, with newlines where needed.

// CHECK: #line
// CHECK: #pragma warning(push)
// CHECK: extern "C" {
// CHECK: #line
// CHECK: #pragma warning(push)
// CHECK:  int foo() { return 0; } }
// CHECK: #pragma warning(pop)
#define A(X) extern "C" { __pragma(warning(push)) \
  int X() { return 0; } \
}
#define B(X) A(X)
#pragma warning(push)
B(foo)
#pragma warning(pop)

#define pragma_L _Pragma(L"GCC diagnostic push")
#define pragma_u8 _Pragma(u8"pack(1)")
#define pragma_u _Pragma(u"GCC diagnostic pop")
#define pragma_U _Pragma(U"comment(lib, \"libfoo\")")
#define pragma_R _Pragma(R"(clang diagnostic ignored "-Wunused")")
#define pragma_UR _Pragma(UR"(clang diagnostic error "-Wunused")")
#define pragma_hello _Pragma(u8R"x(message R"y("Hello", world!)y")x")
// CHECK: int n =
// CHECK: #pragma GCC diagnostic push
// CHECK: #pragma pack(1)
// CHECK: #pragma GCC diagnostic pop
// CHECK: #pragma comment(lib, "libfoo")
// CHECK: #pragma clang diagnostic ignored "-Wunused"
// CHECK: #pragma clang diagnostic error "-Wunused"
// CHECK: #pragma message("\042Hello\042, world!")
// CHECK: 0;
int n = pragma_L pragma_u8 pragma_u pragma_U pragma_R pragma_UR pragma_hello 0;

#pragma warning(disable : 1 2L 3U ; error : 4 5 6 ; suppress : 7 8 9)
// CHECK: #pragma warning(disable: 1 2 3)
// CHECK: #line [[@LINE-2]]
// CHECK: #pragma warning(error: 4 5 6)
// CHECK: #line [[@LINE-4]]
// CHECK: #pragma warning(suppress: 7 8 9)

#pragma warning(push)
#pragma warning(push, 1L)
#pragma warning(push, 4U)
#pragma warning(push, 0x1)
#pragma warning(push, 03)
#pragma warning(push, 0b10)
#pragma warning(push, 1i8)
// CHECK: #pragma warning(push)
// CHECK: #pragma warning(push, 1)
// CHECK: #pragma warning(push, 4)
// CHECK: #pragma warning(push, 1)
// CHECK: #pragma warning(push, 3)
// CHECK: #pragma warning(push, 2)
// CHECK: #pragma warning(push, 1)
