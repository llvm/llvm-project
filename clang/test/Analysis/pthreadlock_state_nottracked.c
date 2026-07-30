// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.PthreadLock,debug.ExprInspection 2>&1 %s | FileCheck %s

#include "Inputs/system-header-simulator-for-pthread-lock.h"

#define NULL 0

void clang_analyzer_printState(void);

void test(pthread_mutex_t *mtx) {
  int ret = pthread_mutex_destroy(mtx);
  clang_analyzer_printState();
  // CHECK:    { "checker": "alpha.core.PthreadLockBase", "messages": [
  // CHECK-NEXT:      "Mutex states:",
  // CHECK-NEXT:      "SymRegion{reg_$[[REG:[0-9]+]]<pthread_mutex_t * mtx>}: not tracked, possibly destroyed",
  // CHECK-NEXT:      "Mutexes in unresolved possibly destroyed state:",
  // CHECK-NEXT:      "SymRegion{reg_$[[REG]]<pthread_mutex_t * mtx>}: conj_$
  // CHECK-NEXT:      ""
  // CHECK-NEXT:    ]}
  if (ret)
    return;
  pthread_mutex_init(mtx, NULL);
}
