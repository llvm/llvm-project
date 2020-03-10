// RUN: %check_clang_tidy %s portability-restrict-system-includes %t \
// RUN:     -- -config="{CheckOptions: [{key: portability-restrict-system-includes.Includes, value: '-*,stdio.h'}]}"

// Test allow-list functionality: disallow all but stdio.h.

#include <stdio.h>
#include <stdlib.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include stdlib.h not allowed
#include <string.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include string.h not allowed
