// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' --experimental-header-filter-matching %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-CLI --allow-empty
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='.*' -exclude-header-filter='confusable_header\.h' %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-EXCLUDE
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='.*' -exclude-header-filter='confusable_header\.h' -config='{ExperimentalHeaderFilterMatching: true}' %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-CONFIG --allow-empty
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' -line-filter='[{"name":"experimental-header-filter-matching.cpp","lines":[[1,20]]}]' %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-LINE-FILTER
// RUN: clang-tidy -checks='-*,misc-confusable-identifiers' -header-filter='does-not-match' -line-filter='[{"name":"experimental-header-filter-matching.cpp","lines":[[1,20]]}]' --experimental-header-filter-matching %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-LINE-FILTER-MATCHING --allow-empty
//
// Positive control: main-file diagnostics must still fire with matching enabled.
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='does-not-match' --experimental-header-filter-matching %s -- -I %S/Inputs/experimental-header-filter-matching 2>&1 | FileCheck %s --check-prefix=CHECK-MAIN-FILE

#include "confusable_header.h"

int lO = 1;
// CHECK-DEFAULT: :[[@LINE-1]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-CLI-NOT: warning:
// CHECK-EXCLUDE: :[[@LINE-3]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-CONFIG-NOT: warning:
// CHECK-LINE-FILTER: :[[@LINE-5]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-LINE-FILTER-MATCHING-NOT: warning:

class A { A(int); };
// CHECK-MAIN-FILE: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit{{.*}} [google-explicit-constructor]
