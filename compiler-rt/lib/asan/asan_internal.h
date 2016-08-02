//===-- asan_internal.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header which defines various general utilities.
//===----------------------------------------------------------------------===//
#ifndef ASAN_INTERNAL_H
#define ASAN_INTERNAL_H

#include "asan_flags.h"
#include "asan_interface_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_libc.h"

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
# error "The AddressSanitizer run-time should not be"
        " instrumented by AddressSanitizer"
#endif

// Build-time configuration options.

// If set, asan will intercept C++ exception api call(s).
#ifndef ASAN_HAS_EXCEPTIONS
# define ASAN_HAS_EXCEPTIONS 1
#endif

// If set, values like allocator chunk size, as well as defaults for some flags
// will be changed towards less memory overhead.
#ifndef ASAN_LOW_MEMORY
#if SANITIZER_WORDSIZE == 32
#  define ASAN_LOW_MEMORY 1
#else
#  define ASAN_LOW_MEMORY 0
# endif
#endif

#ifndef ASAN_DYNAMIC
# ifdef PIC
#  define ASAN_DYNAMIC 1
# else
#  define ASAN_DYNAMIC 0
# endif
#endif

// All internal functions in asan reside inside the __asan namespace
// to avoid namespace collisions with the user programs.
// Separate namespace also makes it simpler to distinguish the asan run-time
// functions from the instrumented user code in a profile.
namespace __asan {

class AsanThread;
using __sanitizer::StackTrace;

void AsanInitFromRtl();

// asan_rtl.cc
void NORETURN ShowStatsAndAbort();

// asan_malloc_linux.cc / asan_malloc_mac.cc
void ReplaceSystemMalloc();

// asan_linux.cc / asan_mac.cc / asan_win.cc
void *AsanDoesNotSupportStaticLinkage();
void AsanCheckDynamicRTPrereqs();
void AsanCheckIncompatibleRT();

void AsanOnDeadlySignal(int, void *siginfo, void *context);

void ReadContextStack(void *context, uptr *stack, uptr *ssize);
void StopInitOrderChecking();

// Wrapper for TLS/TSD.
void AsanTSDInit(void (*destructor)(void *tsd));
void *AsanTSDGet();
void AsanTSDSet(void *tsd);
void PlatformTSDDtor(void *tsd);

void AppendToErrorMessageBuffer(const char *buffer);

void *AsanDlSymNext(const char *sym);

void ReserveShadowMemoryRange(uptr beg, uptr end, const char *name);

// Platform-specific options.
#if SANITIZER_MAC
bool PlatformHasDifferentMemcpyAndMemmove();
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE \
    (PlatformHasDifferentMemcpyAndMemmove())
#else
# define PLATFORM_HAS_DIFFERENT_MEMCPY_AND_MEMMOVE true
#endif  // SANITIZER_MAC

// Add convenient macro for interface functions that may be represented as
// weak hooks.
#define ASAN_MALLOC_HOOK(ptr, size) \
  if (&__sanitizer_malloc_hook) __sanitizer_malloc_hook(ptr, size)
#define ASAN_FREE_HOOK(ptr) \
  if (&__sanitizer_free_hook) __sanitizer_free_hook(ptr)
#define ASAN_ON_ERROR() \
  if (&__asan_on_error) __asan_on_error()

extern int asan_inited;
// Used to avoid infinite recursion in __asan_init().
extern bool asan_init_is_running;
extern void (*death_callback)(void);

// These magic values are written to shadow for better error reporting.
const int kAsanHeapLeftRedzoneMagic = 0xfa;
const int kAsanHeapRightRedzoneMagic = 0xfb;
const int kAsanHeapFreeMagic = 0xfd;
const int kAsanStackLeftRedzoneMagic = 0xf1;
const int kAsanStackMidRedzoneMagic = 0xf2;
const int kAsanStackRightRedzoneMagic = 0xf3;
const int kAsanStackPartialRedzoneMagic = 0xf4;
const int kAsanStackAfterReturnMagic = 0xf5;
const int kAsanInitializationOrderMagic = 0xf6;
const int kAsanUserPoisonedMemoryMagic = 0xf7;
const int kAsanContiguousContainerOOBMagic = 0xfc;
const int kAsanStackUseAfterScopeMagic = 0xf8;
const int kAsanGlobalRedzoneMagic = 0xf9;
const int kAsanInternalHeapMagic = 0xfe;
const int kAsanArrayCookieMagic = 0xac;
const int kAsanIntraObjectRedzone = 0xbb;
const int kAsanAllocaLeftMagic = 0xca;
const int kAsanAllocaRightMagic = 0xcb;

static const uptr kCurrentStackFrameMagic = 0x41B58AB3;
static const uptr kRetiredStackFrameMagic = 0x45E0360E;

}  // namespace __asan

#endif  // ASAN_INTERNAL_H
