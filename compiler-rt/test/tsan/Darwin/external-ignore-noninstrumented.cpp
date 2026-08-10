// RUN: basename %t-lib.dylib | tr -d '\n' > %t.basename
// RUN: %clangxx_tsan -shared %p/external-lib.cpp -fno-sanitize=thread -DUSE_TSAN_CALLBACKS \
// RUN:   -o %t-lib.dylib -install_name @rpath/%{readfile:%t.basename}

// RUN: basename %t-module.dylib | tr -d '\n' > %t.basename
// RUN: %clangxx_tsan -shared %p/external-noninstrumented-module.cpp %t-lib.dylib -fno-sanitize=thread \
// RUN:   -o %t-module.dylib -install_name @rpath/%{readfile:%t.basename}

// RUN: %clangxx_tsan %s %t-module.dylib -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>

extern "C" void NonInstrumentedModule();
int main(int argc, char *argv[]) {
  NonInstrumentedModule();
  fprintf(stderr, "Done.\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: Done.
