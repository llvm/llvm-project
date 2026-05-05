// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' --experimental-header-filter-matching %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-MATCHING --allow-empty
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' --experimental-header-filter-matching %s -- -DEXPAND_IN_MAIN -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-MAIN-EXPANSION
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='macro_def_header\.h' --experimental-header-filter-matching %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-DEFINITION-ONLY --allow-empty

#include "macro_def_header.h"

#ifndef EXPAND_IN_MAIN
#include "macro_expansion_header.h"
#else
DEFINE_CONFUSABLE(l0)
#endif

int lO = 1;
// CHECK-DEFAULT: :[[@LINE-1]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-MATCHING-NOT: warning:
// CHECK-MAIN-EXPANSION: :[[@LINE-3]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-DEFINITION-ONLY-NOT: warning:
