// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.PthreadLock,debug.ExprInspection 2>&1 %s | FileCheck %s

#include "Inputs/system-header-simulator-for-pthread-lock.h"

#define NULL 0

void clang_analyzer_printState();

pthread_mutex_t mtx;

void test() {
  clang_analyzer_printState();
  // CHECK:    "checker_messages": null

  pthread_mutex_init(&mtx, NULL);
  clang_analyzer_printState();
  // CHECK:    { "checker": "alpha.core.PthreadLockBase", "messages": [
  // CHECK-NEXT:      "Mutex states:",
  // CHECK-NEXT:      "mtx: unlocked",
  // CHECK-NEXT:      ""
  // CHECK-NEXT:    ]}

  pthread_mutex_lock(&mtx);
  clang_analyzer_printState();
  // CHECK:    { "checker": "alpha.core.PthreadLockBase", "messages": [
  // CHECK-NEXT:      "Mutex states:",
  // CHECK-NEXT:      "mtx: locked",
  // CHECK-NEXT:      "Mutex lock order:",
  // CHECK-NEXT:      "mtx",
  // CHECK-NEXT:      ""
  // CHECK-NEXT:    ]}

  pthread_mutex_unlock(&mtx);
  clang_analyzer_printState();
  // CHECK:    { "checker": "alpha.core.PthreadLockBase", "messages": [
  // CHECK-NEXT:      "Mutex states:",
  // CHECK-NEXT:      "mtx: unlocked",
  // CHECK-NEXT:      ""
  // CHECK-NEXT:    ]}

  int ret = pthread_mutex_destroy(&mtx);
  clang_analyzer_printState();
  // CHECK:    { "checker": "alpha.core.PthreadLockBase", "messages": [
  // CHECK-NEXT:      "Mutex states:",
  // CHECK-NEXT:      "mtx: unlocked, possibly destroyed",
  // CHECK-NEXT:      "Mutexes in unresolved possibly destroyed state:",
  // CHECK-NEXT:      "mtx: conj_$
  // CHECK-NEXT:      ""
  // CHECK-NEXT:    ]}

  if (ret)
    return;

  clang_analyzer_printState();
  // CHECK:    { "checker": "alpha.core.PthreadLockBase", "messages": [
  // CHECK-NEXT:      "Mutex states:",
  // CHECK-NEXT:      "mtx: destroyed",
  // CHECK-NEXT:      ""
  // CHECK-NEXT:    ]}
}
