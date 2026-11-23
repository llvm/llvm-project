// REQUIRES: x86-registered-target

/// -falign-loops=N emits per-loop !{!"llvm.loop.align", i32 N} metadata.
// RUN: %clang_cc1 -triple=x86_64 -emit-llvm %s -falign-loops=8 -O -o - | FileCheck %s --check-prefix=MD8
// RUN: %clang_cc1 -triple=x86_64 -emit-llvm %s -falign-loops=32 -O -o - | FileCheck %s --check-prefix=MD32

/// End-to-end: the metadata still lowers to .p2align in the backend. The
/// backend takes max(target preferred, metadata), so use values >= the x86
/// default (16) to get an unambiguous alignment.
// RUN: %clang_cc1 -triple=x86_64 -S %s -falign-loops=32 -O -o - | FileCheck %s --check-prefix=ASM32
// RUN: %clang_cc1 -triple=x86_64 -S %s -falign-loops=64 -O -o - | FileCheck %s --check-prefix=ASM64

// MD8: !{!"llvm.loop.align", i32 8}
// MD32: !{!"llvm.loop.align", i32 32}

// ASM32-LABEL: foo:
// ASM32: .p2align 5
// ASM64-LABEL: foo:
// ASM64: .p2align 6

void bar(void);
void foo(void) {
  for (int i = 0; i < 64; ++i)
    bar();
}

/// A source-level [[clang::code_align]] takes precedence over -falign-loops.
/// The attribute value 16 (not the flag's 32) proves the attribute wins; 16 has
/// no other source in this module.
// RUN: %clang_cc1 -triple=x86_64 -emit-llvm %s -falign-loops=32 -O -o - | FileCheck %s --check-prefix=OVERRIDE

// OVERRIDE-LABEL: @baz
// OVERRIDE: br {{.*}}!llvm.loop
// OVERRIDE: !{!"llvm.loop.align", i32 16}

void baz(void) {
  [[clang::code_align(16)]]
  for (int i = 0; i < 64; ++i)
    bar();
}
