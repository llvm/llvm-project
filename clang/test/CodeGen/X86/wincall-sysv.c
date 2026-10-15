// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -S -o - %s | FileCheck %s --check-prefix=ASM

// The wincall calling convention is accepted on any x86-64 target, not only on
// the x86_64apx Windows ones.  On such a target the Windows argument/return
// classification has to be used, just as it already is for
// __attribute__((ms_abi)): X86_64ABIInfo::computeInfo has to delegate to
// WinX86_64ABIInfo for CC_WinCall as well.

struct c24 {
  long a, b, c;
};
struct c40 {
  long a, b, c, d, e;
};
struct empty {};

__attribute__((wincall)) struct c24 ret24(void) { return (struct c24){1, 2, 3}; }
// IR-LABEL: define dso_local x86_wincallcc { i64, i64, i64 } @ret24(
// IR-NOT: sret
// The 24-byte record is coerced to its three eight-byte parts, like the GCC
// wincall port, and comes back in RAX, RDX and RCX.

__attribute__((wincall)) struct c40 ret40(void) {
  return (struct c40){1, 2, 3, 4, 5};
}
// IR-LABEL: define dso_local x86_wincallcc void @ret40(ptr {{.*}}sret(%struct.c40)

__attribute__((wincall)) void take_empty(struct empty e, int x) { (void)e; }
// IR-LABEL: define dso_local x86_wincallcc void @take_empty(i32 noundef %x)

// The convention implies the APX egpr feature, so the fifth integer argument is
// passed in R16 and the compiler no longer aborts in LiveVariables.
__attribute__((wincall)) long sum5(long a, long b, long c, long d, long e) {
  return a + b + c + d + e;
}
// ASM-LABEL: sum5:
// ASM: %r16
