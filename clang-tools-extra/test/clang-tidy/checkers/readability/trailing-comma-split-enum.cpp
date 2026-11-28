// RUN: %check_clang_tidy -std=c++98-or-later %s readability-trailing-comma %t -- \
// RUN:   -- -I %S/Inputs/trailing-comma

// The enum's opening brace is in the header and its closing brace is here.
#include "split-enum.h"
  SplitB
};
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: enum should have a trailing comma
// CHECK-FIXES: SplitB,
// CHECK-FIXES-NEXT: };
