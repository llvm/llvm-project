// RUN: %check_clang_tidy %s llvmlibc-restrict-system-libc-headers %t \
// RUN:   -- -- -isystem %S/Inputs/system \
// RUN:   -resource-dir %S/Inputs/resource

#include <math.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include math.h not allowed
#include <time.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include time.h not allowed
#include "locale.h"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include locale.h not allowed
#include "stdatomic.h"
#include <stdatomic.h>
// Compiler provided headers should not throw warnings.
