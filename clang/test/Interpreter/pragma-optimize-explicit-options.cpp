// REQUIRES: host-supports-jit
//
// An option set at startup keeps its value when '#pragma clang repl optimize'
// changes the level. -fno-inline-functions marks every function noinline.
//
// RUN: cat %s | clang-repl -Xcc -O2 -Xcc -fno-inline-functions \
// RUN:   -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

extern "C" int f_o2(int x) { return x; }
// CHECK:      ; Function Attrs: {{.*}}noinline
// CHECK-NOT:  optnone
// CHECK-SAME: {{$}}
// CHECK-NEXT: define {{.*}}i32 @f_o2(

#pragma clang repl optimize(O3)
extern "C" int f_o3(int x) { return x; }
// CHECK:      ; Function Attrs: {{.*}}noinline
// CHECK-NOT:  optnone
// CHECK-SAME: {{$}}
// CHECK-NEXT: define {{.*}}i32 @f_o3(

#pragma clang repl optimize(O0)
extern "C" int f_o0(int x) { return x; }
// CHECK:      ; Function Attrs: {{.*}}noinline {{.*}}optnone
// CHECK-NEXT: define {{.*}}i32 @f_o0(

#pragma clang repl optimize(O2)
extern "C" int f_back(int x) { return x; }
// CHECK:      ; Function Attrs: {{.*}}noinline
// CHECK-NOT:  optnone
// CHECK-SAME: {{$}}
// CHECK-NEXT: define {{.*}}i32 @f_back(
