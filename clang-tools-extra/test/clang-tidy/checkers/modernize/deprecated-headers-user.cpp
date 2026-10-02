// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-deprecated-headers %t -- -extra-arg-before=-iquote%S/Inputs/deprecated-headers/user -extra-arg-before=-isystem%S/Inputs/deprecated-headers

#include "assert.h"

#include <assert.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]
// CHECK-FIXES: #include <cassert>

int user_header = USER_ASSERT_H;
