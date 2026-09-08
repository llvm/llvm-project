// RUN: not %clang_cc1 -std=c++26 -fherbceptions -emit-obj -o /dev/null %s 2>&1 | FileCheck %s

// A noexcept function must not call a throws function without an explicit
// 'try { } catch throws' block. The bare call auto-propagates the error to
// the caller, which is impossible for a function that cannot return a
// herbception.

inline constexpr void throwing() throws {}

inline constexpr void non_throwing_noexcept() noexcept {
  // CHECK: error: call to 'throws' function in a non-'throws' function must be handled by an enclosing 'try { } catch throws' block, or mark the calling function as 'throws' so herbceptions can propagate
  throwing();
}

int main() {
  non_throwing_noexcept();
  return 0;
}
