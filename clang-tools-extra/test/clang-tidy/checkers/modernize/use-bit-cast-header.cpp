// RUN: %check_clang_tidy -std=c++20-or-later \
// RUN:   -check-header %S/Inputs/use-bit-cast/header.h \
// RUN:   %s modernize-use-bit-cast %t -- --header-filter=.* \
// RUN:   -- -I%S/Inputs/use-bit-cast

#include <cstring>

// CHECK-FIXES: #include <bit>

// Include twice to verify that the guarded header is diagnosed and fixed once.
#include "header.h"
#include "header.h"

void main_file_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning [modernize-use-bit-cast]
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}
