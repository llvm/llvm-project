// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -emit-module %t/module.modulemap -fmodule-name=B -o %t/cache/B.pcm \
// RUN:   -fmodules-single-module-parse-mode 2>&1 | FileCheck %s

// Modules are not imported.
// CHECK-NOT: A.h:1:2: error: unreachable

// Headers belonging to this module are included.
// CHECK:     B2.h:2:2: warning: success

// Non-modular headers are included.
// CHECK:     T.h:2:2: warning: success

// No branches are entered for #if UNDEFINED.
// CHECK-NOT: B1.h:6:2: error: unreachable
// CHECK-NOT: B1.h:8:2: error: unreachable
// CHECK-NOT: B1.h:10:2: error: unreachable

// No branches are entered for #ifdef UNDEFINED.
// CHECK-NOT: B1.h:14:2: error: unreachable
// CHECK-NOT: B1.h:16:2: error: unreachable

// No branches are entered for #ifndef UNDEFINED.
// CHECK-NOT: B1.h:20:2: error: unreachable
// CHECK-NOT: B1.h:22:2: error: unreachable

// No error messages are emitted for UNDEFINED_FUNCTION_LIKE().
// CHECK-NOT: B1.h:25:2: error: unreachable

// The correct branch is entered for #if DEFINED.
// CHECK:     B1.h:32:3: warning: success
// CHECK-NOT: B1.h:34:3: error: unreachable
// CHECK-NOT: B1.h:36:3: error: unreachable

// Headers belonging to this module are included.
// CHECK:     B2.h:2:2: warning: success

//--- module.modulemap
module A { header "A.h" }
module B {
  header "B1.h"
  header "B2.h"
}
//--- A.h
#error unreachable
//--- B1.h
#include "A.h"
#include "B2.h"
#include "T.h"

#if UNDEFINED
# error unreachable
#elif UNDEFINED2
# error unreachable
#else
# error unreachable
#endif

#ifdef UNDEFINED
# error unreachable
#else
# error unreachable
#endif

#ifndef UNDEFINED
# error unreachable
#else
# error unreachable
#endif

#if UNDEFINED_FUNCTION_LIKE()
#endif

#define DEFINED_1 1
#define DEFINED_2 1

#if DEFINED_1
# warning success
#elif DEFINED_2
# error unreachable
#else
# error unreachable
#endif
//--- B2.h
// Headers belonging to this module are included.
#warning success
//--- T.h
// Non-modular headers are included.
#warning success
