// RUN: %clang_cc1 -w -fblocks -triple i386-apple-darwin9 -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

// CHECK-LABEL: define void @f0(%struct.s0* byval align 4)
// CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %{{.*}}, i8* align 4 %{{.*}}, i32 16, i1 false)
// CHECK: }
struct s0 { long double a; };
void f0(struct s0 a0) {
  extern long double f0_g0;
  f0_g0 = a0.a;
}
