// RUN: rm -rf %t
// Build PCH using A, with adjacent private module APrivate, which winds up being implicitly referenced
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -emit-pch -o %t-A.pch %s -Wprivate-module -DNO_AT_IMPORT
// Use the PCH with no explicit way to resolve APrivate, still pick it up by automatic second-chance search for "A" with "Private" appended
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -include-pch %t-A.pch %s -fsyntax-only -Wprivate-module -DNO_AT_IMPORT

// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -emit-pch -o %t-A.pch %s -Wprivate-module -DUSE_AT_IMPORT_PRIV
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -include-pch %t-A.pch %s -fsyntax-only -Wprivate-module -DUSE_AT_IMPORT_PRIV

// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -emit-pch -o %t-A.pch %s -Wprivate-module -DUSE_AT_IMPORT_BOTH
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -include-pch %t-A.pch %s -fsyntax-only -Wprivate-module -DUSE_AT_IMPORT_BOTH

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#ifdef NO_AT_IMPORT
#import "A/aprivate.h"
#endif

#ifdef USE_AT_IMPORT_PRIV
@import A_Private;
#endif

#ifdef USE_AT_IMPORT_BOTH
@import A;
@import A_Private;
#endif

const int *y = &APRIVATE;

#endif
