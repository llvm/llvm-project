//===-- scudo_platform.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo platform specific definitions.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_PLATFORM_H_
#define SCUDO_PLATFORM_H_

#include "sanitizer_common/sanitizer_allocator.h"

#if !SANITIZER_LINUX && !SANITIZER_FUCHSIA
# error "The Scudo hardened allocator is not supported on this platform."
#endif

#define SCUDO_TSD_EXCLUSIVE_SUPPORTED (!SANITIZER_ANDROID && !SANITIZER_FUCHSIA)

#ifndef SCUDO_TSD_EXCLUSIVE
// SCUDO_TSD_EXCLUSIVE wasn't defined, use a default TSD model for the platform.
# if SANITIZER_ANDROID || SANITIZER_FUCHSIA
// Android and Fuchsia use a pool of TSDs shared between threads.
#  define SCUDO_TSD_EXCLUSIVE 0
# elif SANITIZER_LINUX && !SANITIZER_ANDROID
// Non-Android Linux use an exclusive TSD per thread.
#  define SCUDO_TSD_EXCLUSIVE 1
# else
#  error "No default TSD model defined for this platform."
# endif  // SANITIZER_ANDROID || SANITIZER_FUCHSIA
#endif  // SCUDO_TSD_EXCLUSIVE

// If the exclusive TSD model is chosen, make sure the platform supports it.
#if SCUDO_TSD_EXCLUSIVE && !SCUDO_TSD_EXCLUSIVE_SUPPORTED
# error "The exclusive TSD model is not supported on this platform."
#endif

// Maximum number of TSDs that can be created for the Shared model.
#ifndef SCUDO_SHARED_TSD_POOL_SIZE
# define SCUDO_SHARED_TSD_POOL_SIZE 32U
#endif  // SCUDO_SHARED_TSD_POOL_SIZE

namespace __scudo {

#if SANITIZER_CAN_USE_ALLOCATOR64
# if defined(__aarch64__) && SANITIZER_ANDROID
const uptr AllocatorSize = 0x2000000000ULL;  // 128G.
typedef VeryCompactSizeClassMap SizeClassMap;
# elif defined(__aarch64__)
const uptr AllocatorSize = 0x10000000000ULL;  // 1T.
typedef CompactSizeClassMap SizeClassMap;
# else
const uptr AllocatorSize = 0x40000000000ULL;  // 4T.
typedef CompactSizeClassMap SizeClassMap;
# endif
#else
# if SANITIZER_ANDROID
static const uptr RegionSizeLog = 19;
typedef VeryCompactSizeClassMap SizeClassMap;
# else
static const uptr RegionSizeLog = 20;
typedef CompactSizeClassMap SizeClassMap;
# endif
#endif  // SANITIZER_CAN_USE_ALLOCATOR64

}  // namespace __scudo

#endif // SCUDO_PLATFORM_H_
