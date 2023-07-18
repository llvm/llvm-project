// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm \
// RUN:   -fexceptions -fobjc-exceptions \
// RUN:   -o %t %s
// RUN: FileCheck < %t %s
//
// <rdar://problem/7471679> [irgen] [eh] Exception code built with clang (x86_64) crashes

// Check that we don't emit unnecessary personality function references.
struct t0_A { t0_A(); };
struct t0_B { t0_A a; };

// CHECK: define {{.*}} @_Z2t0v(){{.*}} {
// CHECK-NOT: objc_personality
// CHECK: }
t0_B& t0() {
 static t0_B x;
 return x;
}
