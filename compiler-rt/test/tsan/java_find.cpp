// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "java.h"

int const kHeapSize = 1024 * 1024;

static void verify_find(jptr from, jptr to, jptr expected_addr,
                        jptr expected_size) {
  jptr addr = from;
  jptr size = __tsan_java_find(&addr, to);
  if (expected_size) {
    if (!size) {
      fprintf(stderr, "FAILED: range: [%p..%p): found nothing\n", (void *)from,
              (void *)to);
      return;
    } else if (expected_size != size) {
      fprintf(stderr, "FAILED: range: [%p..%p): wrong size, %lu instead of %lu\n",
              (void *)from, (void *)to, size, expected_size);
      return;
    }
  } else if (size) {
    fprintf(stderr,
            "FAILED: range [%p..%p): did not expect to find anything here\n",
            (void *)from, (void *)to);
    return;
  } else {
    return;
  }
  if (expected_addr != addr) {
    fprintf(
        stderr,
        "FAILED: range [%p..%p): expected to find object at %p, found at %p\n",
        (void *)from, (void *)to, (void *)expected_addr, (void *)addr);
  }
}

int main() {
  const jptr jheap = (jptr)malloc(kHeapSize + 8) + 8;
  const jptr jheap_end = jheap + kHeapSize;
  __tsan_java_init(jheap, kHeapSize);
  const jptr addr1 = jheap;
  const int size1 = 16;
  __tsan_java_alloc(jheap, size1);

  const jptr addr2 = addr1 + size1;
  const int size2 = 32;
  __tsan_java_alloc(jheap + size1, size2);

  const jptr addr3 = addr2 + size2;
  const int size3 = 1024;
  __tsan_java_alloc(jheap + size1 + size2, size3);

  const jptr addr4 = addr3 + size3;

  verify_find(jheap, jheap_end, addr1, size1);
  verify_find(jheap + 8, jheap_end, addr2, size2);
  verify_find(addr2 + 8, jheap_end, addr3, size3);
  verify_find(addr3 + 8, jheap_end, 0, 0);

  __tsan_java_move(addr2, addr4, size2);
  verify_find(jheap + 8, jheap_end, addr3, size3);
  verify_find(addr3 + 8, jheap_end, addr4, size2);
  verify_find(addr4 + 8, jheap_end, 0, 0);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: FAILED
// CHECK: DONE
