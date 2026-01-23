// FIXME: These test cases currently crash during codegen, despite initializers
// for CLEs being constant.
// RUN: not --crash %clang_cc1 -verify -std=c++20 -emit-llvm %s -o -
// expected-no-diagnostics
namespace case1 {
struct RR { int&& r; };
struct Z { RR* x; };
constinit Z z = { (RR[1]){1} };
}


namespace case2 {
struct RR { int r; };
struct Z { int x; const RR* y; int z; };
inline int f() { return 0; }
Z z2 = { 10, (const RR[1]){__builtin_constant_p(z2.x)}, z2.y->r+f() };
}
