// Make sure dlerror is not classified as a leak even if we use dynamic TLS.
// This is currently not implemented, so this test is XFAIL.

// Android HWAsan does not support LSan.
// UNSUPPORTED: android

// RUN: %clangxx_hwasan -O0 %s -o %t && HWASAN_OPTIONS=detect_leaks=1 %run %t

#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sanitizer/hwasan_interface.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// musl only has  128 keys
constexpr auto kKeys = 100;

int main(int argc, char **argv) {
  __hwasan_enable_allocator_tagging();
  // Exhaust static TLS slots to force use of dynamic TLS.
  pthread_key_t keys[kKeys];
  for (int i = 0; i < kKeys; ++i) {
    assert(pthread_key_create(&keys[i], nullptr) == 0);
  }
  void *o = dlopen("invalid_file_name.so", 0);
  const char *err = dlerror();
  for (int i = 0; i < kKeys; ++i) {
    assert(pthread_key_delete(keys[i]) == 0);
  }
  return 0;
}
