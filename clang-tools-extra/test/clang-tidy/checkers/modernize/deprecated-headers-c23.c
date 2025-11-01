// RUN: %check_clang_tidy -std=c23-or-later %s modernize-deprecated-headers %t -- -extra-arg-before=-isystem%S/Inputs/deprecated-headers

#include <stdalign.h> // <stdalign.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdalign.h' has no effect since C23; consider removing it
// CHECK-FIXES: // <stdalign.h>
#include <stdbool.h> // <stdbool.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect since C23; consider removing it
// CHECK-FIXES: // <stdbool.h>
#include <stdnoreturn.h> // <stdnoreturn.h>
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdnoreturn.h' has no effect since C23; consider removing it
// CHECK-FIXES: // <stdnoreturn.h>

#include <stdio.h> // OK, not deprecated

#include "stdalign.h" // "stdalign.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdalign.h' has no effect since C23; consider removing it
// CHECK-FIXES: // "stdalign.h"
#include "stdbool.h" // "stdbool.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdbool.h' has no effect since C23; consider removing it
// CHECK-FIXES: // "stdbool.h"
#include "stdnoreturn.h" // "stdnoreturn.h"
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: including 'stdnoreturn.h' has no effect since C23; consider removing it
// CHECK-FIXES: // "stdnoreturn.h"

#include "stdio.h" // OK, not deprecated
