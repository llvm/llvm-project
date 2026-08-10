// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:   -target-feature +sme2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:   -target-feature +sme2 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:   -target-feature +sme2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

volatile int value;

__attribute__((noinline)) static void was_inlined(void) { ++value; }

void normal(void) { was_inlined(); }
void compatible(void) __arm_streaming_compatible { was_inlined(); }
void streaming(void) __arm_streaming { was_inlined(); }
__arm_new("za") void new_za(void) { was_inlined(); }
__arm_new("zt0") void new_zt0(void) { was_inlined(); }

// CIR-LABEL: cir.func{{.*}} @normal_caller()
// LLVM-LABEL: define{{.*}} void @normal_caller()
void normal_caller(void) {
  [[clang::always_inline]] normal();
  // CIR: cir.call @normal() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM-NOT: call void @normal()
  // LLVM: call void @was_inlined()

  [[clang::always_inline]] compatible();
  // CIR: cir.call @compatible() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM-NOT: call void @compatible()
  // LLVM: call void @was_inlined()

  [[clang::always_inline]] streaming();
  // CIR: cir.call @streaming()
  // CIR-NOT: inline_kind
  // LLVM: call void @streaming()

  [[clang::always_inline]] new_za();
  // CIR: cir.call @new_za()
  // CIR-NOT: inline_kind
  // LLVM: call void @new_za()

  [[clang::always_inline]] new_zt0();
  // CIR: cir.call @new_zt0()
  // CIR-NOT: inline_kind
  // CIR: cir.return
  // LLVM: call void @new_zt0()
  // LLVM: ret void
}

// CIR-LABEL: cir.func{{.*}} @compatible_caller()
// LLVM-LABEL: define{{.*}} void @compatible_caller()
void compatible_caller(void) __arm_streaming_compatible {
  [[clang::always_inline]] normal();
  // CIR: cir.call @normal()
  // CIR-NOT: inline_kind
  // LLVM: call void @normal()

  [[clang::always_inline]] compatible();
  // CIR: cir.call @compatible() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM-NOT: call void @compatible()
  // LLVM: call void @was_inlined()

  [[clang::always_inline]] streaming();
  // CIR: cir.call @streaming()
  // CIR-NOT: inline_kind
  // CIR: cir.return
  // LLVM: call void @streaming()
  // LLVM: ret void
}

// CIR-LABEL: cir.func{{.*}} @streaming_caller()
// LLVM-LABEL: define{{.*}} void @streaming_caller()
void streaming_caller(void) __arm_streaming {
  [[clang::always_inline]] normal();
  // CIR: cir.call @normal()
  // CIR-NOT: inline_kind
  // LLVM: call void @normal()

  [[clang::always_inline]] compatible();
  // CIR: cir.call @compatible() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM-NOT: call void @compatible()
  // LLVM: call void @was_inlined()

  [[clang::always_inline]] streaming();
  // CIR: cir.call @streaming() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM-NOT: call void @streaming()
  // LLVM: call void @was_inlined()
  // LLVM: ret void
}
