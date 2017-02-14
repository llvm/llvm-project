//=-- lsan_common_mac.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Implementation of common leak checking functionality. Darwin-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "lsan_common.h"

#if CAN_SANITIZE_LEAKS && SANITIZER_MAC

#include <pthread.h>

namespace __lsan {

static pthread_key_t key;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

static void make_tls_key() { CHECK_EQ(pthread_key_create(&key, NULL), 0); }

static int *get_tls_val(bool allocate) {
  pthread_once(&key_once, make_tls_key);

  int *ptr = (int *)pthread_getspecific(key);
  if (ptr == NULL && allocate) {
    ptr = (int *)InternalAlloc(sizeof(*ptr));
    *ptr = 0;
    pthread_setspecific(key, ptr);
  }

  return ptr;
}

bool DisabledInThisThread() {
  int *disable_counter = get_tls_val(false);
  return disable_counter ? *disable_counter > 0 : false;
}

void DisableInThisThread() {
  int *disable_counter = get_tls_val(true);

  ++*disable_counter;
}

void EnableInThisThread() {
  int *disable_counter = get_tls_val(true);
  if (*disable_counter == 0) {
    DisableCounterUnderflow();
  }
  --*disable_counter;
}

void InitializePlatformSpecificModules() {
  CHECK(0 && "unimplemented");
}

// Scans global variables for heap pointers.
void ProcessGlobalRegions(Frontier *frontier) {
  CHECK(0 && "unimplemented");
}

void ProcessPlatformSpecificAllocations(Frontier *frontier) {
  CHECK(0 && "unimplemented");
}

void DoStopTheWorld(StopTheWorldCallback callback, void *argument) {
  CHECK(0 && "unimplemented");
}

} // namespace __lsan

#endif // CAN_SANITIZE_LEAKS && SANITIZER_MAC
