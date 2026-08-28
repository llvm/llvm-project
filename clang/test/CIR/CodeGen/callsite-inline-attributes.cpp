// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void callee();

void (*fptr)(void) = &callee;

void caller() {
  // CIR-LABEL: cir.func{{.*}}@_Z6callerv()
 
  [[clang::always_inline]]
  callee();
  // CIR: cir.call @_Z6calleev() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM: call void @_Z6calleev() #[[ALWAYSINLINE:.*]]
  [[clang::noinline]]
  callee();
  // CIR: cir.call @_Z6calleev() {inline_kind = #cir.inline_kind<no_inline>}
  // LLVM: call void @_Z6calleev() #[[NOINLINE:.*]]

  [[clang::always_inline]]
  fptr();
  // CIR: cir.call %{{.*}}() {inline_kind = #cir.inline_kind<always_inline>}
  // LLVM: call void %{{.*}}() #[[ALWAYSINLINE]]
  [[clang::noinline]]
  fptr();
  // CIR: cir.call %{{.*}}() {inline_kind = #cir.inline_kind<no_inline>}
  // LLVM: call void %{{.*}}() #[[NOINLINE]]

  [[clang::always_inline]]
  {
    callee();
    // CIR: cir.call @_Z6calleev() {inline_kind = #cir.inline_kind<always_inline>}
    // LLVM: call void @_Z6calleev() #[[ALWAYSINLINE]]
    fptr();
    // CIR: cir.call %{{.*}}() {inline_kind = #cir.inline_kind<always_inline>}
    // LLVM: call void %{{.*}}() #[[ALWAYSINLINE]]
  }

  [[clang::noinline]]
  {
    callee();
    // CIR: cir.call @_Z6calleev() {inline_kind = #cir.inline_kind<no_inline>}
    // LLVM: call void @_Z6calleev() #[[NOINLINE]]
    fptr();
    // CIR: cir.call %{{.*}}() {inline_kind = #cir.inline_kind<no_inline>}
    // LLVM: call void %{{.*}}() #[[NOINLINE]]
  }

  [[clang::noinline]]
  {
    [[clang::always_inline]]
    callee();
    // CIR: cir.call @_Z6calleev() {inline_kind = #cir.inline_kind<always_inline>}
    // LLVM: call void @_Z6calleev() #[[ALWAYSINLINE]]
  }
}

// LLVM: attributes #[[ALWAYSINLINE]] = { alwaysinline }
// LLVM: attributes #[[NOINLINE]] = { noinline }
