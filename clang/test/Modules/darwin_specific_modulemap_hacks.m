// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -isystem %S/Inputs/System/usr/include -triple x86_64-apple-darwin10 %s -verify -fsyntax-only
// expected-no-diagnostics

@import Darwin.C.excluded; // no error, header is implicitly 'textual'
@import Tcl.Private;       // no error, header is implicitly 'textual'
@import IOKit.avc;         // no error, cplusplus requirement removed

#if defined(DARWIN_C_EXCLUDED)
#error assert.h should be textual
#elif defined(TCL_PRIVATE)
#error tcl-private/header.h should be textual
#endif

#import <assert.h>
#import <tcl-private/header.h>

#if !defined(DARWIN_C_EXCLUDED)
#error assert.h missing
#elif !defined(TCL_PRIVATE)
#error tcl-private/header.h missing
#endif
