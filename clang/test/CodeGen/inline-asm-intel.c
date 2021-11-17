// REQUIRES: x86-registered-target

/// Accept intel inline asm but write it out as att:
// RUN: %clang_cc1 -triple i386-unknown-linux -mllvm -x86-asm-syntax=att -inline-asm=intel -Werror -target-feature +hreset -target-feature +pconfig -target-feature +sgx -ffreestanding -O0 -S %s -o - | FileCheck --check-prefix=ATT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -mllvm -x86-asm-syntax=att -inline-asm=intel -Werror -target-feature +hreset -target-feature +pconfig -target-feature +sgx -ffreestanding -O0 -S %s -o - | FileCheck --check-prefix=ATT %s

/// Accept intel inline asm and write it out as intel:
// RUN: %clang_cc1 -triple i386-unknown-linux -mllvm -x86-asm-syntax=intel -inline-asm=intel -Werror -target-feature +hreset -target-feature +pconfig -target-feature +sgx -ffreestanding -O0 -S %s -o - | FileCheck  --check-prefix=INTEL %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -mllvm -x86-asm-syntax=intel -inline-asm=intel -Werror -target-feature +hreset -target-feature +pconfig -target-feature +sgx -ffreestanding -O0 -S %s -o - | FileCheck  --check-prefix=INTEL %s

/// Check MS compat mode (_MSC_VER defined). The driver always picks intel
/// output in that mode, so test only that.
// RUN: %clang_cc1 -triple i386-pc-win32 -mllvm -x86-asm-syntax=intel -inline-asm=intel -Werror -target-feature +hreset -target-feature +pconfig -target-feature +sgx -ffreestanding -O0 -S %s -o - -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 | FileCheck  --check-prefix=INTEL %s
// RUN: %clang_cc1 -triple x86_64-pc-win32 -mllvm -x86-asm-syntax=intel -inline-asm=intel -Werror -target-feature +hreset -target-feature +pconfig -target-feature +sgx -ffreestanding -O0 -S %s -o - -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 | FileCheck  --check-prefix=INTEL %s

// Test that intrinsics headers still work with -masm=intel.
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

void f() {
  // Intrinsic headers contain macros and inline functions.
  // Inline assembly in both are checked only when they are
  // referenced, so reference a few intrinsics here.
  __SSC_MARK(4);
  int a;
  _hreset(a);
  _pconfig_u32(0, (void*)0);

  _encls_u32(0, (void*)0);
  _enclu_u32(0, (void*)0);
  _enclv_u32(0, (void*)0);
#ifdef _MSC_VER
  __movsb((void*)0, (void*)0, 0);
  __movsd((void*)0, (void*)0, 0);
  __movsw((void*)0, (void*)0, 0);
  __stosb((void*)0, 0, 0);
  __stosd((void*)0, 0, 0);
  __stosw((void*)0, 0, 0);
#ifdef __x86_64__
  __movsq((void*)0, (void*)0, 0);
  __stosq((void*)0, 0, 0);
#endif
  __cpuid((void*)0, 0);
  __cpuidex((void*)0, 0, 0);
  __halt();
  __nop();
  __readmsr(0);
  __readcr3();
  __writecr3(0);

  _InterlockedExchange_HLEAcquire((void*)0, 0);
  _InterlockedExchange_HLERelease((void*)0, 0);
  _InterlockedCompareExchange_HLEAcquire((void*)0, 0, 0);
  _InterlockedCompareExchange_HLERelease((void*)0, 0, 0);
#ifdef __x86_64__
  _InterlockedExchange64_HLEAcquire((void*)0, 0);
  _InterlockedExchange64_HLERelease((void*)0, 0);
  _InterlockedCompareExchange64_HLEAcquire((void*)0, 0, 0);
  _InterlockedCompareExchange64_HLERelease((void*)0, 0, 0);
#endif
#endif


  __asm__("mov eax, ebx");
  // ATT: movl %ebx, %eax
  // INTEL: mov eax, ebx

  // Explicitly overriding asm style per block works:
  __asm__(".att_syntax\nmovl %ebx, %eax");
  // ATT: movl %ebx, %eax
  // INTEL: mov eax, ebx

  // The .att_syntax was only scoped to the previous statement.
  // (This is different from gcc, where `.att_syntax` is in
  // effect from that point on, so portable code would want an
  // explicit `.intel_syntax noprefix\n` at the start of this string).
  __asm__("mov eax, ebx");
  // ATT: movl %ebx, %eax
  // INTEL: mov eax, ebx
}

