// REQUIRES: host-supports-jit
//
// '#pragma clang repl optimize(<opt>)' sets the optimization level and size for
// input parsed after it; <opt> is an -O flag spelled without the leading dash.
//
// RUN: cat %s | clang-repl -Xcc -fno-discard-value-names -Xcc -Xclang -Xcc -emit-llvm \
// RUN:   | FileCheck %s

extern "C" int f_o0(int x) { return x; }
// CHECK:      ; Function Attrs: {{.*}}noinline {{.*}}optnone
// CHECK-NEXT: define {{.*}}i32 @f_o0(
// CHECK-NEXT: entry:
// CHECK-NEXT:   %x.addr = alloca i32

#pragma clang repl optimize(O2)
extern "C" int f_o2(int x) { return x; }
// CHECK:      ; Function Attrs:
// CHECK-NOT:  {{minsize|noinline|optnone|optsize}}
// CHECK-SAME: {{$}}
// CHECK-NEXT: define {{.*}}i32 @f_o2(
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 %x

#pragma clang repl optimize(Os)
extern "C" int f_os(int x) { return x; }
// CHECK:      ; Function Attrs:
// CHECK-NOT:  {{minsize|noinline|optnone}}
// CHECK-SAME: optsize
// CHECK-NEXT: define {{.*}}i32 @f_os(
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 %x

#pragma clang repl optimize(Oz)
extern "C" int f_oz(int x) { return x; }
// CHECK:      ; Function Attrs: {{.*}}minsize
// CHECK-NOT:  {{noinline|optnone}}
// CHECK-SAME: optsize
// CHECK-NEXT: define {{.*}}i32 @f_oz(
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 %x

#pragma clang repl optimize(O0)
extern "C" int f_back(int x) { return x; }
// CHECK:      ; Function Attrs:
// CHECK-NOT:  minsize
// CHECK-SAME: noinline {{.*}}optnone
// CHECK-NOT:  optsize
// CHECK-SAME: {{$}}
// CHECK-NEXT: define {{.*}}i32 @f_back(
// CHECK-NEXT: entry:
// CHECK-NEXT:   %x.addr = alloca i32
