// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t

// XFAIL: target-is-mips64el

#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sanitizer/msan_interface.h>

int main(void) {
  unsigned char vec[20];
  int res;
  size_t PS = sysconf(_SC_PAGESIZE);
  void *addr = mmap(nullptr, 20 * PS, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

  __msan_poison(&vec, sizeof(vec));
  res = mincore(addr, 10 * PS, vec);
  assert(res == 0);
  assert(__msan_test_shadow(vec, sizeof(vec)) == 10);

  __msan_poison(&vec, sizeof(vec));
  res = mincore(addr, 10 * PS + 42, vec);
  assert(res == 0);
  assert(__msan_test_shadow(vec, sizeof(vec)) == 11);

  __msan_poison(&vec, sizeof(vec));
  res = mincore(addr, 10 * PS - 1, vec);
  assert(res == 0);
  assert(__msan_test_shadow(vec, sizeof(vec)) == 10);

  __msan_poison(&vec, sizeof(vec));
  res = mincore(addr, 1, vec);
  assert(res == 0);
  assert(__msan_test_shadow(vec, sizeof(vec)) == 1);

  return 0;
}
