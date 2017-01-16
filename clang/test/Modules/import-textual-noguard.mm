// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -fmodules -fimplicit-module-maps -I%S/Inputs/import-textual/M2 -fmodules-cache-path=%t -x objective-c++ -fmodules-local-submodule-visibility %s -verify

#include "A/A.h" // expected-error {{could not build module 'M'}}
#include "B/B.h"

typedef aint xxx;
typedef bint yyy;
