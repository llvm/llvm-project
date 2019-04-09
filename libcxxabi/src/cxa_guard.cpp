//===---------------------------- cxa_guard.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "__cxxabi_config.h"

#include "abort_message.h"
#include "include/atomic_support.h"
#include <__threading_support>

#include <stdint.h>
#include <string.h>

/*
    This implementation must be careful to not call code external to this file
    which will turn around and try to call __cxa_guard_acquire reentrantly.
    For this reason, the headers of this file are as restricted as possible.
    Previous implementations of this code for __APPLE__ have used
    std::__libcpp_mutex_lock and the abort_message utility without problem. This
    implementation also uses std::__libcpp_condvar_wait which has tested
    to not be a problem.
*/

namespace __cxxabiv1
{

namespace
{

enum InitializationResult {
  INIT_COMPLETE,
  INIT_NOT_COMPLETE,
};

#ifdef __arm__
// A 32-bit, 4-byte-aligned static data value. The least significant 2 bits must
// be statically initialized to 0.
typedef uint32_t guard_type;
#else
typedef uint64_t guard_type;
#endif

#if !defined(_LIBCXXABI_HAS_NO_THREADS) && defined(__APPLE__) &&               \
    !defined(__arm__)
// This is a special-case pthread dependency for Mac. We can't pull this
// out into libcxx's threading API (__threading_support) because not all
// supported Mac environments provide this function (in pthread.h). To
// make it possible to build/use libcxx in those environments, we have to
// keep this pthread dependency local to libcxxabi. If there is some
// convenient way to detect precisely when pthread_mach_thread_np is
// available in a given Mac environment, it might still be possible to
// bury this dependency in __threading_support.
#ifndef _LIBCPP_HAS_THREAD_API_PTHREAD
#error "How do I pthread_mach_thread_np()?"
#endif
#define LIBCXXABI_HAS_DEADLOCK_DETECTION
#define LOCK_ID_FOR_THREAD() pthread_mach_thread_np(std::__libcpp_thread_get_current_id())
typedef uint32_t lock_type;
#else
#define LOCK_ID_FOR_THREAD() true
typedef bool lock_type;
#endif

enum class OnRelease : char { UNLOCK, UNLOCK_AND_BROADCAST };

struct GlobalMutexGuard {
  explicit GlobalMutexGuard(const char* calling_func, OnRelease on_release)
      : calling_func(calling_func), on_release(on_release) {
#ifndef _LIBCXXABI_HAS_NO_THREADS
    if (std::__libcpp_mutex_lock(&guard_mut))
      abort_message("%s failed to acquire mutex", calling_func);
#endif
  }

  ~GlobalMutexGuard() {
#ifndef _LIBCXXABI_HAS_NO_THREADS
    if (std::__libcpp_mutex_unlock(&guard_mut))
      abort_message("%s failed to release mutex", calling_func);
    if (on_release == OnRelease::UNLOCK_AND_BROADCAST) {
      if (std::__libcpp_condvar_broadcast(&guard_cv))
        abort_message("%s failed to broadcast condition variable",
                      calling_func);
    }
#endif
  }

  void wait_for_signal() {
#ifndef _LIBCXXABI_HAS_NO_THREADS
    if (std::__libcpp_condvar_wait(&guard_cv, &guard_mut))
      abort_message("%s condition variable wait failed", calling_func);
#endif
  }

private:
  GlobalMutexGuard(GlobalMutexGuard const&) = delete;
  GlobalMutexGuard& operator=(GlobalMutexGuard const&) = delete;

  const char* const calling_func;
  OnRelease on_release;

#ifndef _LIBCXXABI_HAS_NO_THREADS
  static std::__libcpp_mutex_t guard_mut;
  static std::__libcpp_condvar_t guard_cv;
#endif
};

#ifndef _LIBCXXABI_HAS_NO_THREADS
std::__libcpp_mutex_t GlobalMutexGuard::guard_mut = _LIBCPP_MUTEX_INITIALIZER;
std::__libcpp_condvar_t GlobalMutexGuard::guard_cv =
    _LIBCPP_CONDVAR_INITIALIZER;
#endif

struct GuardObject;

/// GuardValue - An abstraction for accessing the various fields and bits of
///   the guard object.
struct GuardValue {
private:
  explicit GuardValue(guard_type v) : value(v) {}
  friend struct GuardObject;

public:
  /// Functions returning the values used to represent the uninitialized,
  /// initialized, and initialization pending states.
  static GuardValue ZERO();
  static GuardValue INIT_COMPLETE();
  static GuardValue INIT_PENDING();

  /// Returns true if the guard value represents that the initialization is
  /// complete.
  bool is_initialization_complete() const;

  /// Returns true if the guard value represents that the initialization is
  /// currently pending.
  bool is_initialization_pending() const;

  /// Returns the lock value for the current guard value.
  lock_type get_lock_value() const;

private:
  // Returns a guard object corresponding to the specified lock value.
  static guard_type guard_value_from_lock(lock_type l);

  // Returns the lock value represented by the specified guard object.
  static lock_type lock_value_from_guard(guard_type g);

private:
  guard_type value;
};

/// GuardObject - Manages correctly reading and writing to the guard object.
struct GuardObject {
  explicit GuardObject(guard_type *g) : guard(g) {}

  /// Load the current value from the guard object.
  GuardValue load() const;

  /// Store the specified value in the guard object.
  void store(GuardValue new_val);

  /// Store the specified value in the guard object and return the previous value.
  GuardValue exchange(GuardValue new_val);

  /// Perform a atomic compare and exchange operation. Return true if
  // desired is written to the guard object.
  bool compare_exchange(GuardValue *expected, GuardValue desired);

private:
  GuardObject(const GuardObject&) = delete;
  GuardObject& operator=(const GuardObject&) = delete;

  guard_type *guard;
};

}  // unnamed namespace

extern "C"
{

_LIBCXXABI_FUNC_VIS int __cxa_guard_acquire(guard_type* raw_guard_object) {
  GlobalMutexGuard gmutex("__cxa_guard_acquire", OnRelease::UNLOCK);
  GuardObject guard(raw_guard_object);
  GuardValue current_value = guard.load();

  if (current_value.is_initialization_complete())
    return INIT_COMPLETE;

  const GuardValue LOCK_ID = GuardValue::INIT_PENDING();
#ifdef LIBCXXABI_HAS_DEADLOCK_DETECTION
   if (current_value.is_initialization_pending() &&
       current_value.get_lock_value() == LOCK_ID.get_lock_value()) {
    abort_message("__cxa_guard_acquire detected deadlock");
  }
#endif
  while (current_value.is_initialization_pending()) {
      gmutex.wait_for_signal();
      current_value = guard.load();
  }
  if (current_value.is_initialization_complete())
    return INIT_COMPLETE;

  guard.store(LOCK_ID);
  return INIT_NOT_COMPLETE;
}

_LIBCXXABI_FUNC_VIS void __cxa_guard_release(guard_type *raw_guard_object) {
  GlobalMutexGuard gmutex("__cxa_guard_release",
                          OnRelease::UNLOCK_AND_BROADCAST);
  GuardObject guard(raw_guard_object);
  guard.store(GuardValue::INIT_COMPLETE());
}

_LIBCXXABI_FUNC_VIS void __cxa_guard_abort(guard_type *raw_guard_object) {
  GlobalMutexGuard gmutex("__cxa_guard_abort", OnRelease::UNLOCK_AND_BROADCAST);
  GuardObject guard(raw_guard_object);
  guard.store(GuardValue::ZERO());
}
}  // extern "C"

//===----------------------------------------------------------------------===//
//                        GuardObject Definitions
//===----------------------------------------------------------------------===//

GuardValue GuardObject::load() const {
  return GuardValue(std::__libcpp_atomic_load(guard));
}

void GuardObject::store(GuardValue new_val) {
  std::__libcpp_atomic_store(guard, new_val.value);
}

GuardValue GuardObject::exchange(GuardValue new_val) {
  return GuardValue(
      std::__libcpp_atomic_exchange(guard, new_val.value, std::_AO_Acq_Rel));
}

bool GuardObject::compare_exchange(GuardValue* expected,
                                   GuardValue desired) {
  return std::__libcpp_atomic_compare_exchange(guard, &expected->value, desired.value,
                                               std::_AO_Acq_Rel, std::_AO_Acquire);
}


//===----------------------------------------------------------------------===//
//                        GuardValue Definitions
//===----------------------------------------------------------------------===//

GuardValue GuardValue::ZERO() { return GuardValue(0); }

GuardValue GuardValue::INIT_COMPLETE() {
  guard_type value = {0};
#ifdef __arm__
  value |= 1;
#else
  char* init_bit = (char*)&value;
  *init_bit = 1;
#endif
  return GuardValue(value);
}

GuardValue GuardValue::INIT_PENDING() {
  return GuardValue(guard_value_from_lock(LOCK_ID_FOR_THREAD()));
}

bool GuardValue::is_initialization_complete() const {
#ifdef __arm__
  return value & 1;
#else
  const char* init_bit = (const char*)&value;
  return *init_bit;
#endif
}

bool GuardValue::is_initialization_pending() const {
  return lock_value_from_guard(value) != 0;
}

lock_type GuardValue::get_lock_value() const {
  return lock_value_from_guard(value);
}

// Create a guard object with the lock set to the specified value.
guard_type GuardValue::guard_value_from_lock(lock_type l) {
#if defined(__APPLE__) && !defined(__arm__)
#if __LITTLE_ENDIAN__
  return static_cast<guard_type>(l) << 32;
#else
  return static_cast<guard_type>(l);
#endif
#else  // defined(__APPLE__) && !defined(__arm__)
  guard_type f = {0};
  memcpy(static_cast<char*>(static_cast<void*>(&f)) + 1, &l, sizeof(lock_type));
  return f;
#endif // defined(__APPLE__) && !defined(__arm__)
}

lock_type GuardValue::lock_value_from_guard(guard_type g) {
#if defined(__APPLE__) && !defined(__arm__)
#if __LITTLE_ENDIAN__
  return static_cast<lock_type>(g >> 32);
#else
  return static_cast<lock_type>(g);
#endif
#else  // defined(__APPLE__) && !defined(__arm__)
  uint8_t guard_bytes[sizeof(guard_type)];
  memcpy(&guard_bytes, &g, sizeof(guard_type));
  return guard_bytes[1] != 0;
#endif // defined(__APPLE__) && !defined(__arm__)
}

}  // __cxxabiv1
