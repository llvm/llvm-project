// RUN: %clang_cc1 -include %s -verify -fsyntax-only -Wthread-safety %s

#ifndef HEADER
#define HEADER

#define LOCKABLE            __attribute__ ((lockable))
#define EXCLUSIVE_LOCK_FUNCTION(...)   __attribute__ ((exclusive_lock_function(__VA_ARGS__)))

class LOCKABLE Mutex{};

template<typename T>
struct lock_guard {
  lock_guard<T>(T) {}
  ~lock_guard<T>() {}
};
template<typename T>
struct unique_lock {
  unique_lock<T>(T) {}
  ~unique_lock<T>() {}
};

template <class T, class... Ts>
void LockMutexes(T &m, Ts &...ms) EXCLUSIVE_LOCK_FUNCTION(m, ms...);

#else

Mutex m0, m1;
void non_local_mutex_held() {
  LockMutexes(m0, m1); // expected-note {{mutex acquired here}} \
  // expected-note {{mutex acquired here}}
} // expected-warning{{mutex 'm0' is still held at the end of function}} \
// expected-warning{{mutex 'm1' is still held at the end of function}}

void no_local_mutex_held_warning() {
  Mutex local_m0;
  Mutex local_m1;
  LockMutexes(local_m0, local_m1);
} // No warnings expected at end of function scope as the mutexes are function local.

void no_local_unique_locks_held_warning() {
  unique_lock<Mutex> ul0(m0);
  unique_lock<Mutex> ul1(m1);
  LockMutexes(ul0, ul1);
} // No warnings expected at end of function scope as the unique_locks held are function local.
#endif
