// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -fmodules -fimplicit-module-maps -I%S/Inputs/import-textual/M -fmodules-cache-path=%t -x objective-c++ %s -verify
// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -fmodules -fimplicit-module-maps -I%S/Inputs/import-textual/M -fmodules-cache-path=%t -x objective-c++ -fmodules-local-submodule-visibility %s -verify

// expected-no-diagnostics

#include "A/A.h"
#include "B/B.h"

typedef aint xxx;
typedef bint yyy;
