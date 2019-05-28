// RUN: %check_clang_tidy -expect-clang-tidy-error %s readability-identifier-naming %t

// This used to cause a null pointer dereference.
auto [left] = right;
// CHECK-MESSAGES: :[[@LINE-1]]:15: error: use of undeclared identifier 'right'
