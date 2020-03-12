// RUN: %check_clang_tidy %s llvmlibc-restrict-system-libc-headers %t \
// RUN:   -- -header-filter=.* \
// RUN:   -- -I %S/Inputs/llvmlibc \
// RUN:   -isystem %S/Inputs/llvmlibc/system \
// RUN:   -resource-dir %S/Inputs/llvmlibc/resource

#include "transitive.h"
// CHECK-MESSAGES: :1:1: warning: system libc header math.h not allowed, transitively included from {{.*}}
