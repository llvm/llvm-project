// Test that sanitizers suppress mlock.
// RUN: %clang  %s -o %t && %run %t

// No shadow, so no need to disable mlock.
// XFAIL: ubsan, lsan

// FIXME: Implement.
// XFAIL: hwasan

#include <assert.h>
#include <sys/mman.h>

int main() {
#if defined(__has_feature)
#  if __has_feature(hwaddress_sanitizer)
  // Don't mlock, it may claim all memory.
  abort();
#  endif
#endif
  assert(0 == mlockall(MCL_CURRENT));
  assert(0 == mlock((void *)0x12345, 0x5678));
  assert(0 == munlockall());
  assert(0 == munlock((void *)0x987, 0x654));
}
