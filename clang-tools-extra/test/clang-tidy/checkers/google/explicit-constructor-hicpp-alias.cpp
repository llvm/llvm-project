// RUN: %check_clang_tidy %s hicpp-explicit-conversions %t

struct A {
  A(int);
  // CHECK-MESSAGES: warning: 'hicpp-explicit-conversions' check is deprecated and will be removed in a future release; consider using 'google-explicit-constructor' instead [clang-tidy-config]
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: single-argument constructors must be marked explicit to avoid unintentional implicit conversions [hicpp-explicit-conversions]
  // CHECK-FIXES: explicit A(int);
};
