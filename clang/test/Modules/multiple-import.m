// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c17 -fmodules-cache-path=%t/no-lsv -fmodules -fimplicit-module-maps -I%t %t/multiple-imports.m -verify
// RUN: %clang_cc1 -std=c17 -fmodules-cache-path=%t/lsv -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -I%t %t/multiple-imports.m -verify

//--- multiple-imports.m
// expected-no-diagnostics
#import <one.h>
#import <assert.h>
void test(void) {
  assert(0);
}

//--- module.modulemap
module Submodules [system] {
  module one {
    header "one.h"
    export *
  }
  module two {
    header "two.h"
    export *
  }
}

module libc [system] {
  textual header "assert.h"
}

//--- one.h
#ifndef one_h
#define one_h
#endif

//--- two.h
#ifndef two_h
#define two_h
#include <assert.h>
#endif

//--- assert.h
#undef assert
#define assert(expression) ((void)0)
