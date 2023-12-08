// RUN: %check_clang_tidy -expect-clang-tidy-error %s cppcoreguidelines-pro-type-member-init %t

struct X {
  X x;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: error: field has incomplete type 'X' [clang-diagnostic-error]
  int a = 10;
};
