// In a non-forking sandbox, we fallback to dladdr(). Test that we provide
// properly demangled C++ names in that case.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=verbosity=2 not %run sandbox-exec -p '(version 1)(allow default)(deny process-fork)' %t 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DLADDR

// sandbox-exec isn't available on iOS
// UNSUPPORTED: ios

#include <stdlib.h>

class MyClass {
 public:
  int my_function(int n) {
    char *x = (char*)malloc(n * sizeof(char));
    free(x);
    return x[5];
    // CHECK-DLADDR: Using dladdr symbolizer
    // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
    // CHECK: {{READ of size 1 at 0x.* thread T0}}
    // CHECK-DLADDR: failed to fork
    // CHECK: {{    #0 0x.* in MyClass::my_function\(int\)}}
    // CHECK: {{freed by thread T0 here:}}
    // CHECK: {{    #0 0x.* in wrap_free}}
    // CHECK: {{    #1 0x.* in MyClass::my_function\(int\)}}
    // CHECK: {{previously allocated by thread T0 here:}}
    // CHECK: {{    #0 0x.* in wrap_malloc}}
    // CHECK: {{    #1 0x.* in MyClass::my_function\(int\)}}
  }
};

int main() {
  MyClass o;
  return o.my_function(10);
}
