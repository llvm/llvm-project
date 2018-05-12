// RUN: %check_clang_tidy %s fuchsia-restrict-system-includes %t \
// RUN:		-- -config="{CheckOptions: [{key: fuchsia-restrict-system-includes.Includes, value: 's.h'}]}" \
// RUN:   -- -std=c++11 -I %S/Inputs/fuchsia-restrict-system-includes -isystem %S/Inputs/fuchsia-restrict-system-includes/system

#include "a.h"

#include <s.h>
#include <t.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include t.h not allowed
// CHECK-FIXES-NOT: #include <t.h>

#include "s.h"
#include "t.h"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include t.h not allowed
// CHECK-FIXES-NOT: #include "t.h"

#define foo <j.h>

#include foo
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include j.h not allowed
// CHECK-FIXES-NOT: #include foo

#/* comment */ include /* comment */ foo
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include j.h not allowed
// CHECK-FIXES-NOT: # /* comment */ include /* comment */ foo
