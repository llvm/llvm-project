//===---------------------------- cxa_guard.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "__cxxabi_config.h"

#include "abort_message.h"
#include "config.h"

#if !LIBCXXABI_HAS_NO_THREADS
#  include <pthread.h>
#endif
#include <stdint.h>

/*
    This implementation must be careful to not call code external to this file
    which will turn around and try to call __cxa_guard_acquire reentrantly.
    For this reason, the headers of this file are as restricted as possible.
    Previous implementations of this code for __APPLE__ have used
    pthread_mutex_lock and the abort_message utility without problem.  This
    implementation also uses pthread_cond_wait which has tested to not be a
    problem.
*/

namespace __cxxabiv1
{

namespace
{

#ifdef __arm__

// A 32-bit, 4-byte-aligned static data value. The least significant 2 bits must
// be statically initialized to 0.
typedef uint32_t guard_type;

// Test the lowest bit.
inline bool is_initialized(guard_type* guard_object) {
    return (*guard_object) & 1;
}

inline void set_initialized(guard_type* guard_object) {
    *guard_object |= 1;
}

#else

typedef uint64_t guard_type;

bool is_initialized(guard_type* guard_object) {
    char* initialized = (char*)guard_object;
    return *initialized;
}

void set_initialized(guard_type* guard_object) {
    char* initialized = (char*)guard_object;
    *initialized = 1;
}

#endif

#if !LIBCXXABI_HAS_NO_THREADS
pthread_mutex_t guard_mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  guard_cv  = PTHREAD_COND_INITIALIZER;
#endif

#if defined(__APPLE__) && !defined(__arm__)

typedef uint32_t lock_type;

#if __LITTLE_ENDIAN__

inline
lock_type
get_lock(uint64_t x)
{
    return static_cast<lock_type>(x >> 32);
}

inline
void
set_lock(uint64_t& x, lock_type y)
{
    x = static_cast<uint64_t>(y) << 32;
}

#else  // __LITTLE_ENDIAN__

inline
lock_type
get_lock(uint64_t x)
{
    return static_cast<lock_type>(x);
}

inline
void
set_lock(uint64_t& x, lock_type y)
{
    x = y;
}

#endif  // __LITTLE_ENDIAN__

#else  // !__APPLE__ || __arm__

typedef bool lock_type;

inline
lock_type
get_lock(uint64_t x)
{
    union
    {
        uint64_t guard;
        uint8_t lock[2];
    } f = {x};
    return f.lock[1] != 0;
}

inline
void
set_lock(uint64_t& x, lock_type y)
{
    union
    {
        uint64_t guard;
        uint8_t lock[2];
    } f = {0};
    f.lock[1] = y;
    x = f.guard;
}

inline
lock_type
get_lock(uint32_t x)
{
    union
    {
        uint32_t guard;
        uint8_t lock[2];
    } f = {x};
    return f.lock[1] != 0;
}

inline
void
set_lock(uint32_t& x, lock_type y)
{
    union
    {
        uint32_t guard;
        uint8_t lock[2];
    } f = {0};
    f.lock[1] = y;
    x = f.guard;
}

#endif  // __APPLE__

}  // unnamed namespace

extern "C"
{

#if LIBCXXABI_HAS_NO_THREADS
_LIBCXXABI_FUNC_VIS int __cxa_guard_acquire(guard_type *guard_object) {
    return !is_initialized(guard_object);
}

_LIBCXXABI_FUNC_VIS void __cxa_guard_release(guard_type *guard_object) {
    *guard_object = 0;
    set_initialized(guard_object);
}

_LIBCXXABI_FUNC_VIS void __cxa_guard_abort(guard_type *guard_object) {
    *guard_object = 0;
}

#else // !LIBCXXABI_HAS_NO_THREADS

_LIBCXXABI_FUNC_VIS int __cxa_guard_acquire(guard_type *guard_object) {
    char* initialized = (char*)guard_object;
    if (pthread_mutex_lock(&guard_mut))
        abort_message("__cxa_guard_acquire failed to acquire mutex");
    int result = *initialized == 0;
    if (result)
    {
#if defined(__APPLE__) && !defined(__arm__)
        const lock_type id = pthread_mach_thread_np(pthread_self());
        lock_type lock = get_lock(*guard_object);
        if (lock)
        {
            // if this thread set lock for this same guard_object, abort
            if (lock == id)
                abort_message("__cxa_guard_acquire detected deadlock");
            do
            {
                if (pthread_cond_wait(&guard_cv, &guard_mut))
                    abort_message("__cxa_guard_acquire condition variable wait failed");
                lock = get_lock(*guard_object);
            } while (lock);
            result = !is_initialized(guard_object);
            if (result)
                set_lock(*guard_object, id);
        }
        else
            set_lock(*guard_object, id);
#else  // !__APPLE__ || __arm__
        while (get_lock(*guard_object))
            if (pthread_cond_wait(&guard_cv, &guard_mut))
                abort_message("__cxa_guard_acquire condition variable wait failed");
        result = *initialized == 0;
        if (result)
            set_lock(*guard_object, true);
#endif  // !__APPLE__ || __arm__
    }
    if (pthread_mutex_unlock(&guard_mut))
        abort_message("__cxa_guard_acquire failed to release mutex");
    return result;
}

_LIBCXXABI_FUNC_VIS void __cxa_guard_release(guard_type *guard_object) {
    if (pthread_mutex_lock(&guard_mut))
        abort_message("__cxa_guard_release failed to acquire mutex");
    *guard_object = 0;
    set_initialized(guard_object);
    if (pthread_mutex_unlock(&guard_mut))
        abort_message("__cxa_guard_release failed to release mutex");
    if (pthread_cond_broadcast(&guard_cv))
        abort_message("__cxa_guard_release failed to broadcast condition variable");
}

_LIBCXXABI_FUNC_VIS void __cxa_guard_abort(guard_type *guard_object) {
    if (pthread_mutex_lock(&guard_mut))
        abort_message("__cxa_guard_abort failed to acquire mutex");
    *guard_object = 0;
    if (pthread_mutex_unlock(&guard_mut))
        abort_message("__cxa_guard_abort failed to release mutex");
    if (pthread_cond_broadcast(&guard_cv))
        abort_message("__cxa_guard_abort failed to broadcast condition variable");
}

#endif // !LIBCXXABI_HAS_NO_THREADS

}  // extern "C"

}  // __cxxabiv1
