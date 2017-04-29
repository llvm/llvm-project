// RUN: rm -rf %t
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x c -DINCLUDE %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x objective-c -DINCLUDE %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x c++ -DINCLUDE %s
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x objective-c -DAT_IMPORT=1 %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x objective-c++ -DAT_IMPORT=1 %s
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x c++ -fmodules-ts -DIMPORT=1 %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x objective-c++ -fmodules-ts -DIMPORT=1 %s
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x c -DPRAGMA %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x objective-c -DPRAGMA %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify -x c++ -DPRAGMA %s

// expected-no-diagnostics

// All forms of module import should make both declarations and macros visible.

#if INCLUDE
#include "dummy.h"
#elif AT_IMPORT
@import dummy;
#elif IMPORT
import dummy;
#elif PRAGMA
#pragma clang module import dummy
#endif

#ifndef DUMMY_H
#error "macros not visible"
#endif

void *p = &dummy1;
