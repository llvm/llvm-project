// Tests that doing dfsan_flush() while another thread is executing doesn't
// segfault.
// RUN: %clang_dfsan %s -o %t && %run %t

#include <assert.h>
#include <pthread.h>
#include <sanitizer/dfsan_interface.h>
#include <stdlib.h>

static unsigned char GlobalBuf[4096];
static int ShutDownThread;
static int StartFlush;

// Access GlobalBuf continuously, causing its shadow to be touched as well.
// When main() calls dfsan_flush(), no segfault should be triggered.
static void *accessGlobalInBackground(void *Arg) {
  __atomic_store_n(&StartFlush, 1, __ATOMIC_RELEASE);

  while (!__atomic_load_n(&ShutDownThread, __ATOMIC_ACQUIRE))
    for (unsigned I = 0; I < sizeof(GlobalBuf); ++I)
      ++GlobalBuf[I];

  return NULL;
}

int main() {
  pthread_t Thread;
  pthread_create(&Thread, NULL, accessGlobalInBackground, NULL);
  while (!__atomic_load_n(&StartFlush, __ATOMIC_ACQUIRE))
    ; // Spin

  dfsan_flush();

  __atomic_store_n(&ShutDownThread, 1, __ATOMIC_RELEASE);
  pthread_join(Thread, NULL);
  return 0;
}
