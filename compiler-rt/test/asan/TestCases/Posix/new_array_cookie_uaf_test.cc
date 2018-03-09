// REQUIRES: asan-64-bits
// RUN: %clangxx_asan -O3 %s -o %t
// RUN: %env_asan_opts=poison_array_cookie=1 not %run %t 2>&1  | FileCheck %s --check-prefix=COOKIE
// RUN: %env_asan_opts=poison_array_cookie=0 not %run %t 2>&1  | FileCheck %s --check-prefix=NO_COOKIE

// UNSUPPORTED: ios

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
int dtor_counter;
struct C {
  int x;
  ~C() {
    dtor_counter++;
    fprintf(stderr, "DTOR %d\n", dtor_counter);
  }
};

__attribute__((noinline)) void Delete(C *c) { delete[] c; }
__attribute__((no_sanitize_address)) void Write42ToCookie(C *c) {
  long *p = reinterpret_cast<long*>(c);
  p[-1] = 42;
}

int main(int argc, char **argv) {
  C *buffer = new C[argc];
  delete [] buffer;
  Write42ToCookie(buffer);
  delete [] buffer;
// COOKIE: DTOR 1
// COOKIE-NOT: DTOR 2
// COOKIE: AddressSanitizer: loaded array cookie from free-d memory
// COOKIE: AddressSanitizer: attempting double-free
// NO_COOKIE: DTOR 1
// NO_COOKIE: DTOR 43
// NO_COOKIE-NOT: DTOR 44
// NO_COOKIE-NOT: AddressSanitizer: loaded array cookie from free-d memory
// NO_COOKIE: AddressSanitizer: attempting double-free

}
