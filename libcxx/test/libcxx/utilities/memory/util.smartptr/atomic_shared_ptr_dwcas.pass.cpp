// REQUIRES: std-at-least-c++20
// REQUIRES: target=x86_64-{{.*}}
// UNSUPPORTED: no-threads
// ADDITIONAL_COMPILE_FLAGS: -march=x86-64-v2

// Verify that with cx16/DWCAS available, is_lock_free() returns true.
// is_always_lock_free remains false (matches standard expectations).

#include <atomic>
#include <cassert>
#include <cstdio>
#include <memory>

int main(int, char**) {
  using A = std::atomic<std::shared_ptr<int>>;

  // is_always_lock_free must remain false to match existing libc++ tests.
  static_assert(!A::is_always_lock_free, "is_always_lock_free should be false");

  A a(std::make_shared<int>(42));
  bool lf = a.is_lock_free();

  // On x86-64 with cx16 (-march=x86-64-v2), DWCAS is available and
  // is_lock_free() should return true.
  std::printf("sizeof=%zu alignof=%zu is_lock_free=%d\n", sizeof(A), alignof(A), (int)lf);

  assert(sizeof(A) == 16);
  assert(alignof(A) == 16);
  assert(lf == true);

  return 0;
}
