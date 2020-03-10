// RUN: %check_clang_tidy %s portability-restrict-system-includes %t \
// RUN:     -- -config="{CheckOptions: [{key: portability-restrict-system-includes.Includes, value: '-*,std*.h'}]}"

// Test glob functionality: disallow all headers except those that match
// pattern "std*.h".

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include string.h not allowed
