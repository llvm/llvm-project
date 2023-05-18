// RUN: %clangxx_msan -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdlib.h>

volatile int z;

int main(void) {
  struct addrinfo *ai;
  struct addrinfo hint;
  int res = getaddrinfo("localhost", NULL, NULL, &ai);
  if (ai) z = 1; // OK
  res = getaddrinfo("localhost", NULL, &hint, &ai);
  // CHECK: Uninitialized bytes in getaddrinfo at offset 0 inside [0x{{.*}}, 48)
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #0 {{.*}} in main {{.*}}getaddrinfo-positive.cpp:[[@LINE-3]]
  return 0;
}
