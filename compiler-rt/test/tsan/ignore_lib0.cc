// RUN: %clangxx_tsan -O1 %s -DLIB -fPIC -fno-sanitize=thread -shared -o %T/libignore_lib0.so
// RUN: %clangxx_tsan -O1 %s -L%T -lignore_lib0 -o %t
// RUN: echo running w/o suppressions:
// RUN: env LD_LIBRARY_PATH=%T${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} %deflake %run %t | FileCheck %s --check-prefix=CHECK-NOSUPP
// RUN: echo running with suppressions:
// RUN: env LD_LIBRARY_PATH=%T${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} %env_tsan_opts=suppressions='%s.supp' %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-WITHSUPP

// Tests that interceptors coming from a library specified in called_from_lib
// suppression are ignored.

// Some aarch64 kernels do not support non executable write pages
// REQUIRES: stable-runtime

// UNSUPPORTED: ios

#ifndef LIB

extern "C" void libfunc();

int main() {
  libfunc();
}

#else  // #ifdef LIB

#include "ignore_lib_lib.h"

#endif  // #ifdef LIB

// CHECK-NOSUPP: WARNING: ThreadSanitizer: data race
// CHECK-NOSUPP: OK

// CHECK-WITHSUPP-NOT: WARNING: ThreadSanitizer: data race
// CHECK-WITHSUPP: OK

