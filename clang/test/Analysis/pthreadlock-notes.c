// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=alpha.unix.PthreadLock \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

#include "Inputs/system-header-simulator-for-pthread-lock.h"

pthread_mutex_t mtx;

void double_lock(void) {
  pthread_mutex_lock(&mtx);   // expected-note{{Locking 'mtx' here}}
  pthread_mutex_lock(&mtx);   // expected-warning{{This lock has already been acquired}}
                              // expected-note@-1{{This lock has already been acquired}}
}

void double_unlock(void) {
  pthread_mutex_lock(&mtx);   // expected-note{{Locking 'mtx' here}}
  pthread_mutex_unlock(&mtx); // expected-note{{Unlocking 'mtx' here}}
  pthread_mutex_unlock(&mtx); // expected-warning{{This lock has already been unlocked}}
                              // expected-note@-1{{This lock has already been unlocked}}
}

void use_after_destroy(void) {
  pthread_mutex_destroy(&mtx); // expected-note{{Destroying 'mtx' here}}
  pthread_mutex_lock(&mtx);    // expected-warning{{This lock has already been destroyed}}
                               // expected-note@-1{{This lock has already been destroyed}}
}

void double_init(void) {
  pthread_mutex_init(&mtx, 0); // expected-note{{Initializing 'mtx' here}}
  pthread_mutex_init(&mtx, 0); // expected-warning{{This lock has already been initialized}}
                               // expected-note@-1{{This lock has already been initialized}}
}

pthread_mutex_t mtx2;

void lock_order_reversal(void) {
  pthread_mutex_lock(&mtx);    // expected-note{{Locking 'mtx' here}}
  pthread_mutex_lock(&mtx2);   // expected-note{{Locking 'mtx2' here}}
  pthread_mutex_unlock(&mtx);  // expected-warning{{This was not the most recently acquired lock}}
                               // expected-note@-1{{This was not the most recently acquired lock}}
}


void double_lock_via_param(pthread_mutex_t *m) {
  pthread_mutex_lock(m);   // expected-note{{Mutex acquired here}}
  pthread_mutex_lock(m);   // expected-warning{{This lock has already been acquired}}
                           // expected-note@-1{{This lock has already been acquired}}
}
