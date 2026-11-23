// RUN: %check_clang_tidy -std=c++14-or-later \
// RUN:   -check-header %S/Inputs/macro-to-enum/modernize-macro-to-enum.h \
// RUN:   -check-header %S/Inputs/macro-to-enum/modernize-macro-to-enum2.h \
// RUN:   -check-header %S/Inputs/macro-to-enum/modernize-macro-to-enum3.h \
// RUN:   %s modernize-macro-to-enum %t -- \
// RUN:   -- -I%S/Inputs/macro-to-enum -fno-delayed-template-parsing

#define HEADER_MAIN 42
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum [modernize-macro-to-enum]
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'HEADER_MAIN' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: HEADER_MAIN = 42
// CHECK-FIXES-NEXT: };

#include "modernize-macro-to-enum.h"
