// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='wrapper_kept_header\.h' %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='wrapper_kept_header\.h' --experimental-header-filter-matching %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-MATCHING

#include "wrapper_header.h"

namespace ns {
int lO = 1;
// CHECK-DEFAULT: :[[@LINE-1]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-MATCHING: :[[@LINE-2]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
} // namespace ns
