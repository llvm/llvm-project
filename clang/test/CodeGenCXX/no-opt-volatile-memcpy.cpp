// RUN: %clang_cc1 -triple=x86_64-apple-darwin  -emit-llvm -o - %s | FileCheck %s
// rdar://11861085

struct s {
  char filler [128];
  volatile int x;
};

struct s gs;

void foo (void) {
  struct s ls;
  ls = ls;
  gs = gs;
  ls = gs;
}
// CHECK-LABEL: define{{.*}} void @_Z3foov()
// CHECK: %[[LS:.*]] = alloca %struct.s, align 4
// CHECK-NEXT:  call void @llvm.memcpy.{{.*}}(ptr align 4 %[[LS]], ptr align 4 %[[LS]], i64 132, i1 true)
// CHECK-NEXT:  call void @llvm.memcpy.{{.*}}(ptr align 4 @gs, ptr align 4 @gs, i64 132, i1 true)
// CHECK-NEXT:  call void @llvm.memcpy.{{.*}}(ptr align 4 %[[LS]], ptr align 4 @gs, i64 132, i1 true)


struct s1 {
  struct s y;
};

struct s1 s;

void fee (void) {
  s = s;
  s.y = gs;
}
// CHECK-LABEL: define{{.*}} void @_Z3feev()
// CHECK: call void @llvm.memcpy.{{.*}}(ptr align 4 @s, ptr align 4 @s, i64 132, i1 true)
// CHECK-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 @s, ptr align 4 @gs, i64 132, i1 true)

struct d : s1 {
};

d gd;

void gorf(void) {
  gd = gd;
}
// CHECK-LABEL: define{{.*}} void @_Z4gorfv()
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr align 4 @gd, ptr align 4 @gd, i64 132, i1 true)
