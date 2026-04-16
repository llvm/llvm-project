// RUN: %check_clang_tidy -std=c++17-or-later %s hicpp-avoid-c-arrays %t

int Values[4];
// CHECK-MESSAGES: warning: 'hicpp-avoid-c-arrays' check is deprecated and will be removed in a future release; consider using 'modernize-avoid-c-arrays' instead [clang-tidy-config]
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: do not declare C-style arrays, use 'std::array' instead [hicpp-avoid-c-arrays]
