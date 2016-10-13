//===------------------------ threading_support.h -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCXXABI_THREADING_SUPPORT_H
#define _LIBCXXABI_THREADING_SUPPORT_H

#include "__cxxabi_config.h"
#include "config.h"

#ifndef _LIBCXXABI_HAS_NO_THREADS

#if defined(_LIBCXXABI_USE_THREAD_API_PTHREAD)
#include <pthread.h>

#define _LIBCXXABI_THREAD_ABI_VISIBILITY inline _LIBCXXABI_INLINE_VISIBILITY

// Mutex
typedef pthread_mutex_t __libcxxabi_mutex_t;
#define _LIBCXXABI_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_mutex_lock(__libcxxabi_mutex_t *mutex) {
  return pthread_mutex_lock(mutex);
}

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_mutex_unlock(__libcxxabi_mutex_t *mutex) {
  return pthread_mutex_unlock(mutex);
}

// Condition variable
typedef pthread_cond_t __libcxxabi_condvar_t;
#define _LIBCXXABI_CONDVAR_INITIALIZER PTHREAD_COND_INITIALIZER

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_condvar_wait(__libcxxabi_condvar_t *cv,
                             __libcxxabi_mutex_t *mutex) {
  return pthread_cond_wait(cv, mutex);
}

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_condvar_broadcast(__libcxxabi_condvar_t *cv) {
  return pthread_cond_broadcast(cv);
}

// Execute once
typedef pthread_once_t __libcxxabi_exec_once_flag;
#define _LIBCXXABI_EXEC_ONCE_INITIALIZER PTHREAD_ONCE_INIT

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_execute_once(__libcxxabi_exec_once_flag *flag,
                             void (*init_routine)(void)) {
  return pthread_once(flag, init_routine);
}

// Thread id
#if defined(__APPLE__) && !defined(__arm__)
_LIBCXXABI_THREAD_ABI_VISIBILITY
mach_port_t __libcxxabi_thread_get_port()
{
    return pthread_mach_thread_np(pthread_self());
}
#endif

// Thread
typedef pthread_t __libcxxabi_thread_t;

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_thread_create(__libcxxabi_thread_t* __t,
                           void* (*__func)(void*), void* __arg)
{
    return pthread_create(__t, 0, __func, __arg);
}

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_thread_join(__libcxxabi_thread_t* __t)
{
    return pthread_join(*__t, 0);
}

// TLS
typedef pthread_key_t __libcxxabi_tls_key;

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_tls_create(__libcxxabi_tls_key *key,
                           void (*destructor)(void *)) {
  return pthread_key_create(key, destructor);
}

_LIBCXXABI_THREAD_ABI_VISIBILITY
void *__libcxxabi_tls_get(__libcxxabi_tls_key key) {
  return pthread_getspecific(key);
}

_LIBCXXABI_THREAD_ABI_VISIBILITY
int __libcxxabi_tls_set(__libcxxabi_tls_key key, void *value) {
  return pthread_setspecific(key, value);
}
#endif // _LIBCXXABI_USE_THREAD_API_PTHREAD
#endif // !_LIBCXXABI_HAS_NO_THREADS
#endif // _LIBCXXABI_THREADING_SUPPORT_H
