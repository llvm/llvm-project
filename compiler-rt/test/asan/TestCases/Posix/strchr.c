// Test strchr for strict_string_checks=false does not look beyond necessary
// char.
// RUN: %clang_asan %s -o %t
// RUN: %env_asan_opts=strict_string_checks=false %run %t 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
  size_t page_size = sysconf(_SC_PAGE_SIZE);
  size_t size = 2 * page_size;
  char *s = (char *)mmap(0, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(s);
  assert(((uintptr_t)s & (page_size - 1)) == 0);
  memset(s, 'o', size);
  s[size - 1] = 0;

  char *p = s + page_size - 1;
  *p = 'x';

  if (mprotect(p + 1, 1, PROT_NONE))
    return 1;
  char *r = strchr(s, 'x');
  // CHECK: AddressSanitizer: {{SEGV|BUS}} on unknown address
  assert(r == p);

  return 0;
}
