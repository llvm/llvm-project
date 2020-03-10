// RUN: %check_clang_tidy %s portability-restrict-system-includes %t \
// RUN:     -- -config="{CheckOptions: [{key: portability-restrict-system-includes.Includes, value: '*,-stdio.h'}]}"

// Test block-list functionality: allow all but stdio.h.

#include <stdio.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include stdio.h not allowed
#include <stdlib.h>
#include <string.h>
