
// Copy the 'mylib.h' to a directory under the build directory. This is
// required, since the relative order of the emitted diagnostics depends on the
// absolute file paths which is sorted by clang-tidy prior emitting.
//
// RUN: mkdir -p %t/sys && mkdir -p %t/usr \
// RUN:   && cp %S/Inputs/modernize-deprecated-headers/mysystemlib.h %t/sys/mysystemlib.h \
// RUN:   && cp %S/Inputs/modernize-deprecated-headers/mylib.h       %t/usr/mylib.h

// RUN: %check_clang_tidy -std=c++11 %s modernize-deprecated-headers %t \
// RUN:   -check-suffixes=DEFAULT \
// RUN:   --header-filter='.*' --system-headers \
// RUN:   -- -I %t/usr -isystem %t/sys -isystem %S/Inputs/modernize-deprecated-headers

// RUN: %check_clang_tidy -std=c++11 %s modernize-deprecated-headers %t \
// RUN:   -check-suffixes=DEFAULT,CHECK-HEADER-FILE \
// RUN:   -config="{CheckOptions: [{key: modernize-deprecated-headers.CheckHeaderFile, value: 'true'}]}" \
// RUN:   --header-filter='.*' --system-headers \
// RUN:   -- -I %t/usr -isystem %t/sys -isystem %S/Inputs/modernize-deprecated-headers

// REQUIRES: system-linux

#define EXTERN_C extern "C"

extern "C++" {
// We should still have the warnings here.
#include <stdbool.h>
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect in C++; consider removing it [modernize-deprecated-headers]
}

#include <assert.h>
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]

#include <stdbool.h>
// CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect in C++; consider removing it [modernize-deprecated-headers]

#include <mysystemlib.h> // no-warning: Don't warn into system headers.

#include <mylib.h>
// CHECK-MESSAGES-CHECK-HEADER-FILE: mylib.h:1:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]

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
