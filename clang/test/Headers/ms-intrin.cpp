// RUN: %clang_cc1 -triple i386-pc-win32 -target-cpu pentium4 \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple i386-pc-win32 -target-cpu broadwell \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -emit-obj -o /dev/null -Werror \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple x86_64-pc-win32  \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -emit-obj -o /dev/null -Werror \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple thumbv7--windows \
// RUN:     -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror \
// RUN:     -isystem %S/Inputs/include %s

// REQUIRES: x86-registered-target

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

// Use some C++ to make sure we closed the extern "C" brackets.
template <typename T>
void foo(T V) {}

// __asm__ blocks are only checked for inline functions that end up being
// emitted, so call functions with __asm__ blocks to make sure their inline
// assembly parses.
void f() {
  __movsb(0, 0, 0);
  __movsd(0, 0, 0);
  __movsw(0, 0, 0);

  __stosb(0, 0, 0);
  __stosd(0, 0, 0);
  __stosw(0, 0, 0);

#ifdef _M_X64
  __movsq(0, 0, 0);
  __stosq(0, 0, 0);
#endif

  int info[4];
  __cpuid(info, 0);
  __cpuidex(info, 0, 0);
  __halt();
  __readmsr(0);

  // FIXME: Call these in 64-bit too once the intrinsics have been fixed to
  // work there, PR19301
#ifndef _M_X64
  __readcr3();
  __writecr3(0);
#endif

#ifdef _M_ARM
  __dmb(_ARM_BARRIER_ISHST);
#endif
}
