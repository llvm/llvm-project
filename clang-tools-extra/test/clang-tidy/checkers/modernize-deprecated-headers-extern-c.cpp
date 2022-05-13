// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-deprecated-headers %t \
// RUN:   -- -header-filter='.*' -system-headers \
// RUN:      -extra-arg-before=-isystem%S/Inputs/modernize-deprecated-headers

#define EXTERN_C extern "C"

extern "C++" {
// We should still have the warnings here.
#include <stdbool.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect in C++; consider removing it [modernize-deprecated-headers]
}

#include <assert.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]
// CHECK-FIXES: {{^}}#include <cassert>{{$}}

#include <stdbool.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect in C++; consider removing it [modernize-deprecated-headers]

#include <mylib.h>
// CHECK-MESSAGES: mylib.h:1:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]

namespace wrapping {
extern "C" {
#include <assert.h>  // no-warning
#include <mylib.h>   // no-warning
#include <stdbool.h> // no-warning
}
} // namespace wrapping

extern "C" {
namespace wrapped {
#include <assert.h>  // no-warning
#include <mylib.h>   // no-warning
#include <stdbool.h> // no-warning
} // namespace wrapped
}

namespace wrapping {
extern "C" {
namespace wrapped {
#include <assert.h>  // no-warning
#include <mylib.h>   // no-warning
#include <stdbool.h> // no-warning
} // namespace wrapped
}
} // namespace wrapping

EXTERN_C {
#include <assert.h>  // no-warning
#include <mylib.h>   // no-warning
#include <stdbool.h> // no-warning
}
