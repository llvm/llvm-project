// RUN: %check_clang_tidy -check-suffixes=ALL -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "All" \
// RUN:     }}' -- -fexceptions
// RUN: %check_clang_tidy -check-suffixes=UNDEFINED -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "OnlyUndefined" \
// RUN:     }}' -- -fexceptions
// RUN: %check_clang_tidy -check-suffixes=NONE -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "None" \
// RUN:     }}' -- -fexceptions

void unannotated_no_throw_body() {}

void calls_unannotated() noexcept {
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_unannotated' which should not throw exceptions
  // CHECK-MESSAGES-UNDEFINED-NOT: warning:
  // CHECK-MESSAGES-NONE-NOT: warning:
  unannotated_no_throw_body();
}

void extern_declared();

void calls_unknown() noexcept {
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_unknown' which should not throw exceptions
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-2]]:6: warning: an exception may be thrown in function 'calls_unknown' which should not throw exceptions
  // CHECK-MESSAGES-NONE-NOT: warning:
  extern_declared();
}

void calls_unknown_caught() noexcept {
  // CHECK-MESSAGES-ALL-NOT: warning:
  // CHECK-MESSAGES-UNDEFINED-NOT: warning:
  // CHECK-MESSAGES-NONE-NOT: warning:
  try {
    extern_declared();
  } catch(...) {
  }
}

void definitely_nothrow() noexcept {}

void calls_nothrow() noexcept {
  // CHECK-MESSAGES-ALL-NOT: warning:
  // CHECK-MESSAGES-UNDEFINED-NOT: warning:
  // CHECK-MESSAGES-NONE-NOT: warning:
  definitely_nothrow();
}

void nothrow_nobody() throw();

void call() noexcept {
  nothrow_nobody();
}

void explicit_throw() { throw 1; }
void calls_explicit_throw() noexcept {
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_explicit_throw' which should not throw exceptions
  // CHECK-MESSAGES-ALL: :[[@LINE-3]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_throw' here
  // CHECK-MESSAGES-ALL: :[[@LINE+7]]:3: note: frame #1: function 'calls_explicit_throw' calls function 'explicit_throw' here
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-4]]:6: warning: an exception may be thrown in function 'calls_explicit_throw' which should not throw exceptions
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-6]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_throw' here
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE+4]]:3: note: frame #1: function 'calls_explicit_throw' calls function 'explicit_throw' here
  // CHECK-MESSAGES-NONE: :[[@LINE-7]]:6: warning: an exception may be thrown in function 'calls_explicit_throw' which should not throw exceptions
  // CHECK-MESSAGES-NONE: :[[@LINE-9]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_throw' here
  // CHECK-MESSAGES-NONE: :[[@LINE+1]]:3: note: frame #1: function 'calls_explicit_throw' calls function 'explicit_throw' here
  explicit_throw();
}
