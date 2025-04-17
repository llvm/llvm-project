// RUN: %clang_cc1 -triple i386-pc-win32 -target-cpu pentium4 \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple i386-pc-win32 -target-cpu broadwell \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -emit-obj -o /dev/null -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple x86_64-pc-win32  \
// RUN:     -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -emit-obj -o /dev/null -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple thumbv7--windows \
// RUN:     -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple aarch64--windows \
// RUN:     -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

// RUN: %clang_cc1 -triple arm64ec--windows \
// RUN:     -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:     -ffreestanding -fsyntax-only -Werror -Wsystem-headers \
// RUN:     -isystem %S/Inputs/include %s

// REQUIRES: x86-registered-target

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

#ifdef __ARM_ACLE
// arm_acle.h needs some stdint types, but -ffreestanding prevents us from
// getting it from stddef.h.  Work around it with these typedefs.
#ifdef __INT8_TYPE__
typedef __INT8_TYPE__ int8_t;
#endif
typedef unsigned char uint8_t;
#ifdef __INT16_TYPE__
typedef __INT16_TYPE__ int16_t;
typedef unsigned __INT16_TYPE__ uint16_t;
#endif
#ifdef __INT32_TYPE__
typedef __INT32_TYPE__ int32_t;
typedef unsigned __INT32_TYPE__ uint32_t;
#endif
#ifdef __INT64_TYPE__
typedef __INT64_TYPE__ int64_t;
typedef unsigned __INT64_TYPE__ uint64_t;
#endif
#include <arm_acle.h>
#endif

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

  __stosd(0, 0, 0);
  __stosw(0, 0, 0);

#if defined(_M_X64) && !defined(_M_ARM64EC)
  __movsq(0, 0, 0);
  __stosq(0, 0, 0);
#endif

  int info[4];
  __cpuid(info, 0);
  __cpuidex(info, 0, 0);
#if (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(_M_IX86)
  _xgetbv(0);
#endif
  __halt();
  __nop();
  __readmsr(0);

  __readcr3();
  __writecr3(0);

#ifdef _M_ARM
  __dmb(_ARM_BARRIER_ISHST);
#endif

#ifdef _M_ARM64
  __dmb(_ARM64_BARRIER_SY);
#endif
}
