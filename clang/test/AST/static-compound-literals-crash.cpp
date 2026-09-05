// FIXME: This test case currently crashes during codegen, despite the
// initializer for the CLE being constant.
// RUN: not --crash %clang_cc1 -verify -std=c++20 -emit-llvm %s -o -
// expected-no-diagnostics
namespace case1 {
struct RR { int&& r; };
struct Z { RR* x; };
constinit Z z = { (RR[1]){1} };
}
