// Test that we can successfully compile this code, especially under ASAN.
// RUN: %clang_cc1 -emit-llvm -std=c++20 %s -o- | FileCheck %s
struct RR { int r; };
struct Z { int x; const RR* y; int z; };
constinit Z z = { 10, (const RR[1]){__builtin_constant_p(z.x)}, z.y->r };
// Check that we zero-initialize z.y->r.
// CHECK: @.compoundliteral = internal constant [1 x %struct.RR] zeroinitializer
// FIXME: Despite of z.y->r being 0, we evaluate z.z to 1.
// CHECK: global %struct.Z { i32 10, ptr @.compoundliteral, i32 1 }
