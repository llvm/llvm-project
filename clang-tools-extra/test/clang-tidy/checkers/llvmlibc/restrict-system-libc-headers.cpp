// RUN: %check_clang_tidy %s llvmlibc-restrict-system-libc-headers %t \
// RUN:   -- -- -isystem %S/Inputs/system \
// RUN:   -resource-dir %S/Inputs/resource

#include <stdio.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include stdio.h not allowed
#include <stdlib.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include stdlib.h not allowed
#include "string.h"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: system include string.h not allowed
#include "stdatomic.h"
#include <stddef.h>
// Compiler provided headers should not throw warnings.
