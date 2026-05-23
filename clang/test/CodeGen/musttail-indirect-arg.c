// Test that Clang forwards incoming Indirect parameters across musttail calls
// instead of creating a byval-temp alloca that would dangle after the tail call
// deallocates the caller's frame.
//
// Companion to musttail-sret.cpp (commit a96c14eeb8fc): same idea, applied to
// incoming arguments rather than the sret return slot.

// RUN: %clang_cc1 -triple=riscv64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=loongarch64-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON
// RUN: %clang_cc1 -triple=s390x-linux-gnu %s -emit-llvm -O1 -o - | FileCheck %s --check-prefix=COMMON

// A struct large enough to land on the indirect-arg path on RV64 (>2*XLEN=16
// bytes), AArch64 (>16 bytes), LoongArch64, SystemZ.
struct Big {
  unsigned long long a, b, c, d;
};

// Plain forward: caller(B) musttails callee(B). The fix should emit no
// byval-temp alloca; the call should forward the incoming parameter %a.
struct Big C1(struct Big a);
struct Big P1(struct Big a) {
  __attribute__((musttail)) return C1(a);
}
// COMMON-LABEL: define {{.*}} @P1(
// COMMON-NOT: alloca %struct.Big
// COMMON: musttail call {{.*}} @C1({{.*}} %a

// Two indirect args, same forwarding: each forwards its own incoming param.
struct Big C2(struct Big a, struct Big b);
struct Big P2(struct Big a, struct Big b) {
  __attribute__((musttail)) return C2(a, b);
}
// COMMON-LABEL: define {{.*}} @P2(
// COMMON-NOT: alloca %struct.Big
// COMMON: musttail call {{.*}} @C2({{.*}} %a, {{.*}} %b

// Swapped args: caller(a, b) musttails callee(b, a). Each forwarded slot
// must resolve to the correct incoming Argument, not by position.
struct Big C3(struct Big x, struct Big y);
struct Big P3(struct Big a, struct Big b) {
  __attribute__((musttail)) return C3(b, a);
}
// COMMON-LABEL: define {{.*}} @P3(
// COMMON-NOT: alloca %struct.Big
// COMMON: musttail call {{.*}} @C3({{.*}} %b, {{.*}} %a

// Mixed direct + indirect: only the indirect arg is affected by the fix.
struct Big C4(int n, struct Big a);
struct Big P4(int n, struct Big a) {
  __attribute__((musttail)) return C4(n, a);
}
// COMMON-LABEL: define {{.*}} @P4(
// COMMON-NOT: alloca %struct.Big
// COMMON: musttail call {{.*}} @C4({{.*}} %n, {{.*}} %a

// Negative: local source. Caller takes Big a, but musttails with a LOCAL
// Big initialized in caller's frame. The byval-temp must remain because the
// source lives in caller's frame and would dangle if forwarded. The fix
// must NOT engage in this case.
struct Big C5(struct Big a);
struct Big P5(struct Big a) {
  struct Big local = {1, 2, 3, 4};
  __attribute__((musttail)) return C5(local);
}
// COMMON-LABEL: define {{.*}} @P5(
// COMMON: alloca
// COMMON: musttail call {{.*}} @C5(

// Negative: computed value (caller modifies the parameter then musttails).
// The IR will use %a directly (Clang lowers writes through the incoming
// pointer for Indirect params) so the fix does engage on the formal param,
// but a fresh alloca is not created either way -- existing behavior.
struct Big C6(struct Big a);
struct Big P6(struct Big a) {
  a.a += 1;
  __attribute__((musttail)) return C6(a);
}
// COMMON-LABEL: define {{.*}} @P6(
// COMMON-NOT: alloca %struct.Big
// COMMON: musttail call {{.*}} @C6({{.*}} %a

// Non-musttail tail call: the fix must NOT engage. Existing path emits
// the byval-temp as before.
struct Big C7(struct Big a);
struct Big P7(struct Big a) {
  return C7(a);
}
// COMMON-LABEL: define {{.*}} @P7(
// COMMON-NOT: musttail
