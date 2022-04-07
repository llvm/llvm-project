// RUN: %clang_cc1 -no-opaque-pointers -triple s390x-linux-gnu -target-cpu z13 -emit-llvm -o - %s \
// RUN:     | FileCheck %s

struct S0 {
  long f1;
  int f2 : 4;
} d;

#pragma pack(1)
struct S1 {
  struct S0 S0_member;
};

void f(struct S0 arg) {
  arg.f2 = 1;
}

void g(void) {
  struct S1 g;
  // CHECK: alloca %struct.S0, align 8
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 {{.*}}, i8* align 1 {{.*}}, i64 16
  f(g.S0_member);
}
