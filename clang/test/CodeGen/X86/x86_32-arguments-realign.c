// RUN: %clang_cc1 -w -fblocks -triple i386-apple-darwin9 -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

// CHECK-LABEL: define{{.*}} void @f0(ptr noundef byval(%struct.s0) align 4 %0)
// CHECK:   call void @llvm.memcpy.p0.p0.i32(ptr align 16 %{{.*}}, ptr align 4 %{{.*}}, i32 16, i1 false)
// CHECK: }
struct s0 { long double a; };
void f0(struct s0 a0) {
  extern long double f0_g0;
  f0_g0 = a0.a;
}
