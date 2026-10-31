// REQUIRES: host-supports-jit
//
// -emit-llvm prints the IR of each input before running them.
//
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

extern "C" int add(int a, int b) { return a + b; }
// CHECK: define {{.*}}i32 @add(
// CHECK: ret i32

extern "C" int answer = 42;
// CHECK: @answer = {{.*}}i32 42

extern "C" int neg(int a) { return -a; }
// CHECK: define {{.*}}i32 @neg(

extern "C" int r = add(19, 23);
// CHECK: @r = {{.*}}global i32
// CHECK: call {{.*}}i32 @add(
r
// CHECK: (int) 42
