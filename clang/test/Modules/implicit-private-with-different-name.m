// RUN: rm -rf %t

// Build PCH using A, with adjacent private module APrivate, which winds up being implicitly referenced
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-different-name -emit-pch -o %t-A.pch %s

// Use the PCH with no explicit way to resolve PrivateA, still pick it up through MODULE_DIRECTORY reference in PCH control block
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-different-name -include-pch %t-A.pch %s -fsyntax-only

#ifndef HEADER
#define HEADER
#import "A/aprivate.h"
const int *y = &APRIVATE;
#endif
