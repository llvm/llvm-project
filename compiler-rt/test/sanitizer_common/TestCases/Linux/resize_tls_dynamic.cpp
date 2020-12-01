// RUN: %clangxx %s -DBUILD_DSO -fPIC -shared -o %t.so
// RUN: %clangxx --std=c++11 %s -o %t
// RUN: %env_tool_opts=verbosity=2 %run %t 2>&1 | FileCheck %s

// Does not call __tls_get_addr
// UNSUPPORTED: i386-linux

// Do not intercept __tls_get_addr
// UNSUPPORTED: lsan, ubsan, android

#include <string.h>

#ifndef BUILD_DSO

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

char buff[10000];

int main(int argc, char *argv[]) {
  sprintf(buff, "rm -f %s.so.*", argv[0]);
  system(buff);

  void *prev_handle = 0;
  for (int i = 0; i < 300; ++i) {
    sprintf(buff, "cp %s.so %s.so.%d", argv[0], argv[0], i);
    system(buff);

    sprintf(buff, "%s.so.%d", argv[0], i);
    void *handle = dlopen(buff, RTLD_LAZY);
    assert(handle != 0);
    assert(handle != prev_handle);
    prev_handle = handle;

    typedef void (*FnType)(char c);
    ((FnType)dlsym(handle, "StoreToTLS"))(i);
  }
  return 0;
}

#else  // BUILD_DSO
__thread char huge_thread_local_array[1 << 12];

extern "C" void StoreToTLS(char c) {
  memset(huge_thread_local_array, c, sizeof(huge_thread_local_array));
}
#endif // BUILD_DSO

// CHECK:      DTLS_Find [[DTLS:0x[a-f0-9]+]] {{[0-9]+}}
// CHECK-NEXT: DTLS_NextBlock [[DTLS]] 0
// CHECK:      DTLS_Find [[DTLS:0x[a-f0-9]+]] 255
// CHECK-NEXT: DTLS_NextBlock [[DTLS]] 1
// CHECK-NOT:  DTLS_NextBlock