// RUN: %check_clang_tidy -std=c++98 %s hicpp-deprecated-headers %t -- -extra-arg-before=-isystem%S/Inputs/deprecated-headers

#include <assert.h>
// CHECK-MESSAGES: warning: 'hicpp-deprecated-headers' check is deprecated and will be removed in a future release; consider using 'modernize-deprecated-headers' instead [clang-tidy-config]
// CHECK-MESSAGES: :[[@LINE-2]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [hicpp-deprecated-headers]
// CHECK-FIXES: #include <cassert>
