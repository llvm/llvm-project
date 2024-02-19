// RUN: %check_clang_tidy %s portability-restrict-system-includes %t \
// RUN:     -- -config="{CheckOptions: {portability-restrict-system-includes.Includes: '*,-stddef.h'}}" \
// RUN:     -- -isystem %S/Inputs/restrict-system-includes/system

// Test block-list functionality: allow all but stddef.h.

#include <stddef.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include stddef.h not allowed
#include <stdint.h>
#include <float.h>
