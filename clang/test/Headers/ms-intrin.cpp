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
#include <arm_acle.h>
#endif

// Use some C++ to make sure we closed the extern "C" brackets.
template <typename T>
void foo(T V) {}

#if defined(_M_X64) && !defined(_M_ARM64EC)
static_assert(__is_same(decltype(__readcr0()), unsigned __int64), "");
static_assert(__is_same(decltype(__readcr2()), unsigned __int64), "");
static_assert(__is_same(decltype(__readcr3()), unsigned __int64), "");
static_assert(__is_same(decltype(__readcr4()), unsigned __int64), "");
static_assert(__is_same(decltype(__readcr8()), unsigned __int64), "");
static_assert(__is_same(decltype(__readdr(0)), unsigned __int64), "");
static_assert(__is_same(decltype(__readeflags()), unsigned __int64), "");
static_assert(__is_same(decltype(__readmsr(0)), unsigned __int64), "");
static_assert(__is_same(decltype(&__writecr0), void (*)(unsigned __int64)), "");
static_assert(__is_same(decltype(&__writecr2), void (*)(unsigned __int64)), "");
static_assert(__is_same(decltype(&__writecr3), void (*)(unsigned __int64)), "");
static_assert(__is_same(decltype(&__writecr4), void (*)(unsigned __int64)), "");
static_assert(__is_same(decltype(&__writecr8), void (*)(unsigned __int64)), "");
static_assert(__is_same(decltype(&__writedr),
                        void (*)(unsigned int, unsigned __int64)),
              "");
static_assert(__is_same(decltype(&__writeeflags), void (*)(unsigned __int64)),
              "");
static_assert(__is_same(decltype(&__writemsr),
                        void (*)(unsigned long, unsigned __int64)),
              "");
#elif defined(_M_IX86)
static_assert(__is_same(decltype(__readcr0()), unsigned long), "");
static_assert(__is_same(decltype(__readcr2()), unsigned long), "");
static_assert(__is_same(decltype(__readcr3()), unsigned long), "");
static_assert(__is_same(decltype(__readcr4()), unsigned long), "");
static_assert(__is_same(decltype(__readcr8()), unsigned long), "");
static_assert(__is_same(decltype(__readdr(0)), unsigned int), "");
static_assert(__is_same(decltype(__readeflags()), unsigned int), "");
static_assert(__is_same(decltype(__readmsr(0)), unsigned __int64), "");
static_assert(__is_same(decltype(&__writecr0), void (*)(unsigned int)), "");
static_assert(__is_same(decltype(&__writecr2), void (*)(unsigned int)), "");
static_assert(__is_same(decltype(&__writecr3), void (*)(unsigned int)), "");
static_assert(__is_same(decltype(&__writecr4), void (*)(unsigned int)), "");
static_assert(__is_same(decltype(&__writecr8), void (*)(unsigned int)), "");
static_assert(__is_same(decltype(&__writedr),
                        void (*)(unsigned int, unsigned int)),
              "");
static_assert(__is_same(decltype(&__writeeflags), void (*)(unsigned int)), "");
static_assert(__is_same(decltype(&__writemsr),
                        void (*)(unsigned long, unsigned __int64)),
              "");
#endif

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
