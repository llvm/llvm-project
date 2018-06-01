// Test the mmap_limit_mb flag.
//
// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %run %t 20 16
// RUN: %run %t 30 1000000
// RUN: %env_asan_opts=mmap_limit_mb=300 %run %t 20 16
// RUN: %env_asan_opts=mmap_limit_mb=300 %run %t 20 1000000
// RUN: %env_asan_opts=mmap_limit_mb=300 not %run %t 500 16 2>&1 | FileCheck %s
// RUN: %env_asan_opts=mmap_limit_mb=300 not %run %t 500 1000000 2>&1 | FileCheck %s
//
// FIXME: Windows doesn't implement mmap_limit_mb.
// XFAIL: win32

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

int main(int argc, char **argv) {
  assert(argc == 3);
  long total_mb = atoi(argv[1]);
  long allocation_size = atoi(argv[2]);
  fprintf(stderr, "total_mb: %zd allocation_size: %zd\n", total_mb,
          allocation_size);
  std::vector<char *> v;
  for (long total = total_mb << 20; total > 0; total -= allocation_size)
    v.push_back(new char[allocation_size]);
  for (std::vector<char *>::const_iterator it = v.begin(); it != v.end(); ++it)
    delete[](*it);
  fprintf(stderr, "PASS\n");
  // CHECK: total_mmaped{{.*}}mmap_limit_mb
  return 0;
}
