// RUN: %clang_cc1 -triple %itanium_abi_triple -fms-extensions -emit-llvm -o- %s | FileCheck %s

void fa(__unaligned struct A *) {}
// CHECK: define {{(dso_local )?}}void @_Z2faPU11__unaligned1A(

void ga(struct A *, struct A *) {}
// CHECK: define {{(dso_local )?}}void @_Z2gaP1AS0_(

void gb(__unaligned struct A *, struct A *) {}
// CHECK: define {{(dso_local )?}}void @_Z2gbPU11__unaligned1APS_(

void gc(struct A *, __unaligned struct A *) {}
// CHECK: define {{(dso_local )?}}void @_Z2gcP1APU11__unalignedS_(

void gd(__unaligned struct A *, __unaligned struct A *) {}
// CHECK: define {{(dso_local )?}}void @_Z2gdPU11__unaligned1AS1_(

void hb(__unaligned struct A *, __unaligned const struct A *) {}
// CHECK: define {{(dso_local )?}}void @_Z2hbPU11__unaligned1APU11__unalignedKS_(

void ja(__unaligned struct A *, __unaligned struct A *__unaligned *, __unaligned struct A *__unaligned *__unaligned *) {}
// CHECK: define {{(dso_local )?}}void @_Z2jaPU11__unaligned1APU11__unalignedS1_PU11__unalignedS3_(

struct A;
void memptr(void (A::*a)(int) __unaligned) {}
// CHECK: define {{.*}} @_Z6memptrM1AU11__unalignedFviE(

void jb(__unaligned struct A *, __unaligned struct A **, __unaligned struct A *__unaligned *__unaligned *) {}
// CHECK: @_Z2jbPU11__unaligned1APS1_PU11__unalignedPU11__unalignedS1_(

template <typename T, typename Q>
void ta(T &, Q *) {}

void ia(__unaligned struct A &a) {
  ta(a, &a);
}
// CHECK: @_Z2taIU11__unaligned1AS1_EvRT_PT0_(
