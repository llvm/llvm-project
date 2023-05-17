// RUN: %check_clang_tidy -expect-clang-tidy-error %s modernize-use-noexcept %t

// We're not interested in the check issuing a warning here, just making sure
// clang-tidy doesn't assert.
undefined_type doesThrow() throw();
// CHECK-MESSAGES: :[[@LINE-1]]:1: error: unknown type name 'undefined_type' [clang-diagnostic-error]
