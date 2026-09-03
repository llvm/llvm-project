// REQUIRES: host-supports-jit
//
// '#pragma clang repl optimize(<opt>)' sets the optimization level and size for
// input parsed after it; <opt> is an -O flag spelled without the leading dash.
//
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

extern "C" int f_o0() { return 0; }
// CHECK: Function Attrs:{{.*}} optnone
// CHECK-NEXT: define {{.*}} @f_o0()

#pragma clang repl optimize(O2)
extern "C" int f_o2() { return 1; }
// CHECK: define {{.*}} @f_o2()
// CHECK-NOT: optnone

#pragma clang repl optimize(Os)
extern "C" int f_os() { return 2; }
// CHECK: Function Attrs:{{.*}} optsize
// CHECK-NEXT: define {{.*}} @f_os()

#pragma clang repl optimize(Oz)
extern "C" int f_oz() { return 3; }
// CHECK: Function Attrs:{{.*}} minsize
// CHECK-NEXT: define {{.*}} @f_oz()

#pragma clang repl optimize(O0)
extern "C" int f_back() { return 4; }
// CHECK: Function Attrs:{{.*}} optnone
// CHECK-NEXT: define {{.*}} @f_back()
