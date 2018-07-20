// RUN: rm -rf %t

// PR35939: MicrosoftMangle.cpp triggers an assertion failure on this test.
// UNSUPPORTED: system-windows

// RUN: %clang_cc1 \
// RUN:  -I %S/Inputs/odr_hash-Friend \
// RUN:  -emit-obj -o /dev/null \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -std=c++11 -x c++ %s -verify -DTEST1 -fcolor-diagnostics

// RUN: %clang_cc1 \
// RUN:  -I %S/Inputs/odr_hash-Friend \
// RUN:  -emit-obj -o /dev/null \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -std=c++11 -x c++ %s -verify -DTEST2 -fcolor-diagnostics

// RUN: %clang_cc1 \
// RUN:  -I %S/Inputs/odr_hash-Friend \
// RUN:  -emit-obj -o /dev/null \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -std=c++11 -x c++ %s -verify -DTEST3 -fcolor-diagnostics

// RUN: %clang_cc1 \
// RUN:  -I %S/Inputs/odr_hash-Friend \
// RUN:  -emit-obj -o /dev/null \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -std=c++11 -x c++ %s -verify -DTEST3 -fcolor-diagnostics

// RUN: %clang_cc1 \
// RUN:  -I %S/Inputs/odr_hash-Friend \
// RUN:  -emit-obj -o /dev/null \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -std=c++11 -x c++ %s -verify -DTEST3 -fcolor-diagnostics

#if defined(TEST1)
#include "Box.h"
#include "M1.h"
#include "M3.h"
// expected-no-diagnostics
#endif

#if defined(TEST2)
#include "Box.h"
#include "M1.h"
#include "M3.h"
#include "Good.h"
// expected-no-diagnostics
#endif

#if defined(TEST3)
#include "Good.h"
#include "Box.h"
#include "M1.h"
#include "M3.h"
// expected-no-diagnostics
#endif

#if defined(TEST4)
#include "Box.h"
#include "M1.h"
#include "M3.h"
#include "Bad.h"
// expected-error@Bad.h:* {{'Check' has different definitions in different modules; definition in module 'Bad' first difference is function body}}
// expected-note@Box.h:* {{but in 'Box' found a different body}}
#endif

#if defined(TEST5)
#include "Bad.h"
#include "Box.h"
#include "M1.h"
#include "M3.h"
// expected-error@Bad.h:* {{'Check' has different definitions in different modules; definition in module 'Bad' first difference is function body}}
// expected-note@Box.h:* {{but in 'Box' found a different body}}
#endif


void Run() {
  Box<> Present;
}
