// RUN: %check_clang_tidy %s abseil-no-namespace %t -- -- -I %S/Inputs
// RUN: clang-tidy -checks='-*, abseil-no-namespace' -header-filter='.*' %s -- -I %S/Inputs 2>&1 | FileCheck %s

/// Warning will not be triggered on internal Abseil code that is included.
#include "absl/strings/internal-file.h"
// CHECK-NOT: warning:

/// Warning will be triggered on code that is not internal that is included.
#include "absl/external-file.h"
// CHECK: absl/external-file.h:1:11: warning: namespace 'absl' is reserved

namespace absl {}
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: namespace 'absl' is reserved for implementation of the Abseil library and should not be opened in user code [abseil-no-namespace]

namespace absl {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: namespace 'absl'
namespace std {
int i = 5;
}
}

// Things that shouldn't trigger the check
int i = 5;
namespace std {}
