// trylock should not count towards a lock cycle; this test checks that this
// holds for Darwin-specific locks

// RUN: %clang_tsan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <libkern/OSAtomic.h>
#include <libkern/OSSpinLockDeprecated.h>
#include <os/lock.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

long global_variable;

// --- os_unfair_lock ---------------------------------------------------------

os_unfair_lock unfair1 = OS_UNFAIR_LOCK_INIT, unfair2 = OS_UNFAIR_LOCK_INIT;

// Holds unfair1, then *tries* unfair2. Were an edge recorded for the
// successful trylock, this would register unfair1 -> unfair2.
void *UnfairTrylockSecond(void *a) {
  os_unfair_lock_lock(&unfair1);
  if (os_unfair_lock_trylock(&unfair2)) {
    global_variable++;
    os_unfair_lock_unlock(&unfair2);
  }
  os_unfair_lock_unlock(&unfair1);
  return NULL;
}

// Blocking locks in the opposite order register unfair2 -> unfair1; combined
// with a wrongly-recorded unfair1 -> unfair2 edge this would form a cycle.
void *UnfairLockDescending(void *a) {
  os_unfair_lock_lock(&unfair2);
  os_unfair_lock_lock(&unfair1);
  global_variable++;
  os_unfair_lock_unlock(&unfair1);
  os_unfair_lock_unlock(&unfair2);
  return NULL;
}

// --- OSSpinLock -------------------------------------------------------------

#pragma clang diagnostic push // OSSpinLock* deprecation
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

OSSpinLock spin1 = OS_SPINLOCK_INIT, spin2 = OS_SPINLOCK_INIT;

void *SpinTrylockSecond(void *a) {
  OSSpinLockLock(&spin1);
  if (OSSpinLockTry(&spin2)) {
    global_variable++;
    OSSpinLockUnlock(&spin2);
  }
  OSSpinLockUnlock(&spin1);
  return NULL;
}

void *SpinLockDescending(void *a) {
  OSSpinLockLock(&spin2);
  OSSpinLockLock(&spin1);
  global_variable++;
  OSSpinLockUnlock(&spin1);
  OSSpinLockUnlock(&spin2);
  return NULL;
}

#pragma clang diagnostic pop // OSSpinLock* deprecation

static void RunPair(void *(*first)(void *), void *(*second)(void *)) {
  pthread_t t;
  pthread_create(&t, NULL, first, NULL);
  pthread_join(t, NULL);
  pthread_create(&t, NULL, second, NULL);
  pthread_join(t, NULL);
}

int main() {
  global_variable = 0;

  RunPair(UnfairTrylockSecond, UnfairLockDescending);
  fprintf(stderr, "os_unfair_lock done\n");
  // CHECK: os_unfair_lock done

  RunPair(SpinTrylockSecond, SpinLockDescending);
  fprintf(stderr, "OSSpinLock done\n");
  // CHECK: OSSpinLock done

  fprintf(stderr, "global_variable = %ld\n", global_variable);
  // CHECK: global_variable = 4
}
