// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.UnknownAsThrowing": true \
// RUN:     }}' -- -fexceptions

void unannotated_no_throw_body() {}

void calls_unannotated() noexcept {
  // CHECK-MESSAGES-NOT: warning:
  unannotated_no_throw_body();
}

void extern_declared();

void calls_unknown() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_unknown' which should not throw exceptions
  extern_declared();
}

void definitely_nothrow() noexcept {}

void calls_nothrow() noexcept {
  // CHECK-MESSAGES-NOT: warning:
  definitely_nothrow();
}
