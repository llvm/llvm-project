// Test that globals from different shared objects all get registered.

// This source file is compiled into three different source object files. Each
// object file declares a global buffer. The first two are linked together, and
// the third is loaded at runtime. We make sure that out-of-bounds accesses
// are caught for all three buffers.

// RUN: %clang_asan -c -o %t-one.o -DMAIN_FILE %s
// RUN: %clang_asan -c -o %t-two.o -DSECONDARY_FILE %s
// RUN: %clang_asan -o %t %t-one.o %t-two.o
// RUN: %clang_asan -o %t-dynamic.so -shared -fPIC %libdl -DSHARED_LIBRARY_FILE %s
// RUN: not %run %t 1 2>&1 | FileCheck --check-prefix ASAN-CHECK-1 %s
// RUN: not %run %t 2 2>&1 | FileCheck --check-prefix ASAN-CHECK-2 %s
// RUN: not %run %t 3 2>&1 | FileCheck --check-prefix ASAN-CHECK-3 %s

#if MAIN_FILE

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

extern char buffer2[1];
char buffer1[1] = { };

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  if (n == 1) {
    buffer1[argc] = 0;
    // ASAN-CHECK-1: {{0x.* is located 1 bytes .* 'buffer1'}}
  } else if (n == 2) {
    buffer2[argc] = 0;
    // ASAN-CHECK-2: {{0x.* is located 1 bytes .* 'buffer2'}}
  } else if (n == 3) {
    char *libpath;
    asprintf(&libpath, "%s-dynamic.so", argv[0]);
    
    void *handle = dlopen(libpath, RTLD_NOW);
    if (!handle) {
      fprintf(stderr, "dlopen: %s\n", dlerror());
      return 1;
    }
    
    char *buffer = (char *)dlsym(handle, "buffer3");
    if (!buffer) {
      fprintf(stderr, "dlsym: %s\n", dlerror());
      return 1;
    }
    
    buffer[argc] = 0;
    // ASAN-CHECK-3: {{0x.* is located 1 bytes .* 'buffer3'}}
  }
  
  return 0;
}

#elif SECONDARY_FILE

char buffer2[1] = { };

#elif SHARED_LIBRARY_FILE

char buffer3[1] = { };

#endif
