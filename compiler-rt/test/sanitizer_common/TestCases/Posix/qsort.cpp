// RUN: %clangxx -O2 %s -o %t

#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static int compare_ints(const void *a, const void *b) {
  return *(const int *)b - *(const int *)a;
}

int main() {
  std::vector<int> nums(100000);
  for (auto &n : nums)
    n = rand();

  std::vector<int> to_qsort = nums;
  qsort(to_qsort.data(), to_qsort.size(), sizeof(to_qsort[0]), &compare_ints);

  std::sort(nums.begin(), nums.end());

  assert(nums == to_qsort);
}
