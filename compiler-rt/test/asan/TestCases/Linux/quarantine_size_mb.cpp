// Test quarantine_size_mb (and the deprecated quarantine_size)
// RUN: %clangxx_asan  %s -o %t
// RUN: %env_asan_opts=quarantine_size=10485760:verbosity=1:hard_rss_limit_mb=50 %run %t  2>&1 | FileCheck %s  --check-prefix=Q10
// RUN: %env_asan_opts=quarantine_size_mb=10:verbosity=1:hard_rss_limit_mb=50    %run %t  2>&1 | FileCheck %s  --check-prefix=Q10
// RUN: %env_asan_opts=quarantine_size_mb=10:quarantine_size=20:verbosity=1  not %run %t  2>&1 | FileCheck %s  --check-prefix=BOTH
// RUN: %env_asan_opts=quarantine_size_mb=1000:hard_rss_limit_mb=50 not  %run %t          2>&1 | FileCheck %s  --check-prefix=RSS_LIMIT
// RUN: %env_asan_opts=hard_rss_limit_mb=20                         not  %run %t          2>&1 | FileCheck %s  --check-prefix=RSS_LIMIT

// https://github.com/google/sanitizers/issues/981
// UNSUPPORTED: android-26

#include <string.h>
char *g;

static const int kNumAllocs = 1 << 11;
static const int kAllocSize = 1 << 20;

int main() {
  for (int i = 0; i < kNumAllocs; i++) {
    g = new char[kAllocSize];
    memset(g, -1, kAllocSize);
    delete [] (g);
  }
}

// Q10: quarantine_size_mb=10M
// BOTH: please use either 'quarantine_size' (deprecated) or quarantine_size_mb, but not both
// RSS_LIMIT: AddressSanitizer: hard rss limit exhausted
