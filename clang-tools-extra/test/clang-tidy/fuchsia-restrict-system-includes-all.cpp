// RUN: %check_clang_tidy %s fuchsia-restrict-system-includes %t \
// RUN:		-- -config="{CheckOptions: [{key: fuchsia-restrict-system-includes.Includes, value: ''}]}" \
// RUN:   -- -std=c++11 -I %S/Inputs/fuchsia-restrict-system-includes -isystem %S/Inputs/fuchsia-restrict-system-includes/system

#include <cstdlib.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include cstdlib.h not allowed
#include <cstdarg.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include cstdarg.h not allowed
#include <t.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include t.h not allowed
