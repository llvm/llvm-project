// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -I%S/Inputs -verify -x objective-c %t/macro-module-name.m
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -I%S/Inputs -verify -x objective-c %t/macro-at-import.m
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -I%S/Inputs -verify -x objective-c %t/macro-func-import.m

//--- macro-module-name.m
// expected-no-diagnostics
#define M dummy
@import M;

#ifndef DUMMY_H
#error "macros from module not visible after @import with macro module name"
#endif

void *p = &dummy1;

//--- macro-at-import.m
#define imp @import
imp dummy;

#ifdef DUMMY_H
#error "module should not be imported via macro-constructed @import"
#endif

void *p = &dummy1; // expected-error {{use of undeclared identifier 'dummy1'}}

//--- macro-func-import.m
#define IMPORT(X) @import X
IMPORT(dummy);

#ifdef DUMMY_H
#error "module should not be imported via function-like macro @import"
#endif

void *p = &dummy1; // expected-error {{use of undeclared identifier 'dummy1'}}
