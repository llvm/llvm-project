// RUN: %clang_cc1 -triple arm64-windows -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-MSVC

// RUN: not %clang_cc1 -triple arm64-linux -Werror -S -o /dev/null %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LINUX

long test_InterlockedAdd(long volatile *Addend, long Value) {
  return _InterlockedAdd(Addend, Value);
}

long test_InterlockedAdd_constant(long volatile *Addend) {
  return _InterlockedAdd(Addend, -1);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd(i32* %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add i32* %1, i32 %2 seq_cst, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %2
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: implicit declaration of function '_InterlockedAdd'

void check__dmb(void) {
  __dmb(0);
}

// CHECK-MSVC: @llvm.aarch64.dmb(i32 0)
// CHECK-LINUX: error: implicit declaration of function '__dmb'

void check__dsb(void) {
  __dsb(0);
}

// CHECK-MSVC: @llvm.aarch64.dsb(i32 0)
// CHECK-LINUX: error: implicit declaration of function '__dsb'

void check__isb(void) {
  __isb(0);
}

// CHECK-MSVC: @llvm.aarch64.isb(i32 0)
// CHECK-LINUX: error: implicit declaration of function '__isb'

void check__yield(void) {
  __yield();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 1)
// CHECK-LINUX: error: implicit declaration of function '__yield'

void check__wfe(void) {
  __wfe();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 2)
// CHECK-LINUX: error: implicit declaration of function '__wfe'

void check__wfi(void) {
  __wfi();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 3)
// CHECK-LINUX: error: implicit declaration of function '__wfi'

void check__sev(void) {
  __sev();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 4)
// CHECK-LINUX: error: implicit declaration of function '__sev'

void check__sevl(void) {
  __sevl();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 5)
// CHECK-LINUX: error: implicit declaration of function '__sevl'

void check_ReadWriteBarrier() {
  _ReadWriteBarrier();
}

// CHECK-MSVC: fence syncscope("singlethread")
// CHECK-LINUX: error: implicit declaration of function '_ReadWriteBarrier'

unsigned __int64 check__getReg() {
  unsigned volatile __int64 reg;
  reg = __getReg(18);
  reg = __getReg(31);
  return reg;
}

// CHECK-MSVC: call i64 @llvm.read_register.i64(metadata ![[MD2:.*]])
// CHECK-MSVC: call i64 @llvm.read_register.i64(metadata ![[MD3:.*]])
// CHECK-MSVC: ![[MD2]] = !{!"x18"}
// CHECK-MSVC: ![[MD3]] = !{!"sp"}
