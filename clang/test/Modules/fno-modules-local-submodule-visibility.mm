// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t -fsyntax-only -I %t/include -x objective-c++ \
// RUN:   %t/tu.m -verify
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t -fsyntax-only -I %t/include -x objective-c++ \
// RUN:   %t/tu2.m -verify -fmodules-local-submodule-visibility

// RUN: not %clang_cc1 -std=c++20 -fmodules -fcxx-modules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t -fsyntax-only -I %t/include -x objective-c++ \
// RUN:   %t/driver.mm -fno-modules-local-submodule-visibility 2>&1 \
// RUN:   | FileCheck %t/driver.mm -check-prefix=CPP20
// RUN: not %clang_cc1 -std=c++20 -fmodules-ts -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t -fsyntax-only -I %t/include -x objective-c++ \
// RUN:   %t/driver.mm -fno-modules-local-submodule-visibility 2>&1 \
// RUN:   | FileCheck %t/driver.mm -check-prefix=TS

//--- include/module.modulemap

module M {
  module A {
    header "A.h"
    export *
  }
  module B {
    header "B.h"
    export *
  }
}

//--- include/A.h
#define A 1

//--- include/B.h
inline int B = A;

//--- tu.m
@import M;
int i = B;
// expected-no-diagnostics

//--- tu2.m
@import M; // expected-error {{could not build module 'M'}}

//--- driver.mm
// CPP20: error: C++20 modules require the -fmodules-local-submodule-visibility -cc1 option
// TS: error: Modules TS modules require the -fmodules-local-submodule-visibility -cc1 option
