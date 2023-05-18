// RUN: %clangxx_asan -std=c++11 -O0 %s -o %t && %run %t

#include <assert.h>
#include <unistd.h>
#include <sys/mman.h>

int main(void) {
  unsigned char vec[20];
  int res;
  size_t PS = sysconf(_SC_PAGESIZE);
  void *addr = mmap(nullptr, 20 * PS, PROT_READ | PROT_WRITE,
                    MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

  res = mincore(addr, 10 * PS, vec);
  assert(res == 0);
  for (int i = 0; i < 10; ++i)
    assert((vec[i] & 1) == 0);

  for (int i = 0; i < 5; ++i)
    ((char *)addr)[i * PS] = 1;
  res = mincore(addr, 10 * PS, vec);
  assert(res == 0);
  for (int i = 0; i < 10; ++i)
    assert((vec[i] & 1) == (i < 5));

  for (int i = 5; i < 10; ++i)
    ((char *)addr)[i * PS] = 1;
  res = mincore(addr, 10 * PS, vec);
  assert(res == 0);
  for (int i = 0; i < 10; ++i)
    assert((vec[i] & 1) == 1);

  return 0;
}
