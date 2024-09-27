// RUN: %clangxx_nsan -O2 -g %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Check compilation mode that is required to call memcpy/memmove.
// RUN: %clangxx_nsan -fno-builtin -O2 -g %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// This tests a particularaly hard case of memory tracking. stable_sort does
// conditional swaps of pairs of elements with mixed types (int/double).

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes,
                                       size_t bytes_per_line, size_t reserved);

__attribute__((noinline)) void Run(std::vector<int> &indices,
                                   std::vector<double> &values) {
  const auto num_entries = indices.size();
  std::vector<std::pair<int, double>> entries;
  entries.reserve(num_entries);
  for (size_t i = 0; i < num_entries; ++i)
    entries.emplace_back(indices[i], values[i]);

  __nsan_dump_shadow_mem((const char *)&entries[0].second, sizeof(double),
                         sizeof(double), 0);
  __nsan_dump_shadow_mem((const char *)&entries[1].second, sizeof(double),
                         sizeof(double), 0);
  // CHECK: {{.*}}: d0 d1 d2 d3 d4 d5 d6 d7 (1.02800000000000002487)
  // CHECK-NEXT: {{.*}}: d0 d1 d2 d3 d4 d5 d6 d7 (7.95099999999999962341)
  std::stable_sort(
      entries.begin(), entries.end(),
      [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
        return a.first < b.first;
      });
  __nsan_dump_shadow_mem((const char *)&entries[0].second, sizeof(double),
                         sizeof(double), 0);
  __nsan_dump_shadow_mem((const char *)&entries[1].second, sizeof(double),
                         sizeof(double), 0);
  // We make sure that the shadow values have been swapped correctly.
  // CHECK-NEXT: {{.*}}: d0 d1 d2 d3 d4 d5 d6 d7 (7.95099999999999962341)
  // CHECK-NEXT: {{.*}}: d0 d1 d2 d3 d4 d5 d6 d7 (1.02800000000000002487)
}

int main() {
  std::vector<int> indices;
  std::vector<double> values;
  indices.push_back(75);
  values.push_back(1.028);
  indices.push_back(74);
  values.push_back(7.951);
  Run(indices, values);
}
