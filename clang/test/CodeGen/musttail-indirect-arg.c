// RUN: %clang_cc1 -triple=riscv64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=s390x-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON

// Each indirect musttail argument must use its matching incoming slot.

// Plain Indirect-ABI struct on the targets above.
struct Big {
  unsigned long long a, b, c, d;
};

// P1 forwards one incoming argument.
struct Big C1(struct Big a);
struct Big P1(struct Big a) {
  __attribute__((musttail)) return C1(a);
}
// COMMON-LABEL: define {{.*}} @P1(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @C1({{.*}}, ptr {{[^,]*}} %a)

// P2 forwards two distinct incoming arguments.
struct Big C2(struct Big a, struct Big b);
struct Big P2(struct Big a, struct Big b) {
  __attribute__((musttail)) return C2(a, b);
}
// COMMON-LABEL: define {{.*}} @P2(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON-NOT: llvm.memcpy
// COMMON: musttail call {{.*}} @C2({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P3 captures %a before the swap overwrites it.
struct Big C3(struct Big x, struct Big y);
struct Big P3(struct Big a, struct Big b) {
  __attribute__((musttail)) return C3(b, a);
}
// COMMON-LABEL: define {{.*}} @P3(
// COMMON: [[SAVED:%musttail.copy[0-9.a-z]*]] = load {{.*}}, ptr %a,
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{[^,]*}} %b,
// COMMON: store {{.*}} [[SAVED]], ptr %b,
// COMMON: musttail call {{.*}} @C3({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P5 forwards a modified incoming argument.
struct Big C5(struct Big a);
struct Big P5(struct Big a) {
  a.a += 1;
  __attribute__((musttail)) return C5(a);
}
// COMMON-LABEL: define {{.*}} @P5(
// COMMON: add i64 {{.*}}, 1
// COMMON: store i64 {{.*}}, ptr %a
// COMMON: musttail call {{.*}} @C5({{.*}}, ptr {{[^,]*}} %a)

// P6 places the musttail call behind a branch.
struct Big C6(struct Big a, int cond);
struct Big P6(struct Big a, int cond) {
  if (cond)
    __attribute__((musttail)) return C6(a, cond);
  return a;
}
// COMMON-LABEL: define {{.*}} @P6(
// COMMON: br i1 {{.*}}, label %if.end, label %if.then
// COMMON: if.then:
// COMMON: musttail call {{.*}} @C6({{.*}}, ptr {{[^,]*}} %a,

// P7 gives two distinct by-value slots the value from %a.
struct Big C7(struct Big x, struct Big y);
struct Big P7(struct Big a, struct Big b) {
  __attribute__((musttail)) return C7(a, a);
}
// COMMON-LABEL: define {{.*}} @P7(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %b, ptr {{[^,]*}} %a,
// COMMON: musttail call {{.*}} @C7({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P8 copies a local value into the incoming slot.
struct Big C8(struct Big a);
struct Big P8(struct Big a) {
  struct Big local = {1, 2, 3, 4};
  __attribute__((musttail)) return C8(local);
}
// COMMON-LABEL: define {{.*}} @P8(
// COMMON: llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %a, ptr {{.*}}
// COMMON: musttail call {{.*}} @C8({{.*}}, ptr {{[^,]*}} %a)

volatile struct Big volatile_source;
struct Big C_volatile(struct Big a);
struct Big P_volatile(struct Big a) {
  __attribute__((musttail)) return C_volatile(volatile_source);
}
// COMMON-LABEL: define {{.*}} @P_volatile(
// COMMON: [[READ:%[0-9a-z.]+]] = load volatile <4 x i64>, ptr @volatile_source
// COMMON: store <4 x i64> [[READ]], ptr %a
// COMMON: musttail call {{.*}} @C_volatile({{.*}}, ptr {{[^,]*}} %a)

struct Big C_volatile_identity(struct Big a);
struct Big P_volatile_identity(volatile struct Big a) {
  __attribute__((musttail)) return C_volatile_identity(a);
}
// COMMON-LABEL: define {{.*}} @P_volatile_identity(
// COMMON: [[SAME:%[0-9a-z.]+]] = load volatile <4 x i64>, ptr %a
// COMMON: store <4 x i64> [[SAME]], ptr %a
// COMMON: musttail call {{.*}} @C_volatile_identity({{.*}}, ptr {{[^,]*}} %a)

// P9 keeps an ordinary call separate from the musttail path.
struct Big C9(struct Big a);
struct Big P9(struct Big a) {
  return C9(a);
}
// COMMON-LABEL: define {{.*}} @P9(
// COMMON-NOT: musttail

// P10 mixes direct and indirect arguments.
struct Big C10(int x1, struct Big s1, int x2, struct Big s2);
struct Big P10(int x1, struct Big s1, int x2, struct Big s2) {
  __attribute__((musttail)) return C10(x1, s1, x2, s2);
}
// COMMON-LABEL: define {{.*}} @P10(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @C10({{.*}}, i32 {{.*}} %x1, ptr {{[^,]*}} %s1, i32 {{.*}} %x2, ptr {{[^,]*}} %s2)

// P11 forwards arguments that include stack-spilled slots.
struct Big C11(struct Big s1, struct Big s2, struct Big s3, struct Big s4,
               struct Big s5, struct Big s6, struct Big s7, struct Big s8,
               struct Big s9, struct Big s10);
struct Big P11(struct Big a1, struct Big a2, struct Big a3, struct Big a4,
               struct Big a5, struct Big a6, struct Big a7, struct Big a8,
               struct Big a9, struct Big a10) {
  __attribute__((musttail)) return C11(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
// COMMON-LABEL: define {{.*}} @P11(
// COMMON-NOT: = alloca {{.*}}struct.Big
// COMMON: musttail call {{.*}} @C11(
// COMMON-SAME: ptr {{[^,]*}} %a1, ptr {{[^,]*}} %a2, ptr {{[^,]*}} %a3, ptr {{[^,]*}} %a4
// COMMON-SAME: ptr {{[^,]*}} %a5, ptr {{[^,]*}} %a6, ptr {{[^,]*}} %a7, ptr {{[^,]*}} %a8
// COMMON-SAME: ptr {{[^,]*}} %a9, ptr {{[^,]*}} %a10

// P12 forwards an over-aligned struct.
struct __attribute__((aligned(32))) AlignedBig {
  unsigned long long a, b, c, d;
};
struct AlignedBig C12(struct AlignedBig a);
struct AlignedBig P12(struct AlignedBig a) {
  __attribute__((musttail)) return C12(a);
}
// COMMON-LABEL: define {{.*}} @P12(
// COMMON: musttail call {{.*}} @C12({{.*}}, ptr {{[^,]*}} align 32 {{[^,]*}} %a)

// P13 mixes a local source with an incoming parameter.
struct Big C13(struct Big x, struct Big y);
struct Big P13(struct Big a, struct Big b) {
  struct Big local = {1, 2, 3, 4};
  __attribute__((musttail)) return C13(local, a);
}
// COMMON-LABEL: define {{.*}} @P13(
// COMMON-NOT: byval-temp
// COMMON: %musttail.copy{{[0-9.a-z]*}} =
// COMMON: musttail call {{.*}} @C13({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b)

// P17 copies %a independently into two other slots.
struct Big C17(struct Big x, struct Big y, struct Big z);
struct Big P17(struct Big a, struct Big b, struct Big c) {
  __attribute__((musttail)) return C17(a, a, a);
}
// COMMON-LABEL: define {{.*}} @P17(
// COMMON: [[SAVED:%musttail.copy[0-9.a-z]*]] = load {{.*}}, ptr %a,
// COMMON: @llvm.mem{{(cpy|move)}}{{.*}}(ptr {{[^,]*}} %b, ptr {{[^,]*}} %a,
// COMMON: store {{.*}} [[SAVED]], ptr %c,
// COMMON: musttail call {{.*}} @C17({{.*}}, ptr {{[^,]*}} %a, ptr {{[^,]*}} %b, ptr {{[^,]*}} %c)
