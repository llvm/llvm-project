// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_FRAMEWORK=1 -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %S/Inputs/APINotes  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_SEARCH_PATH=1 -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -iapinotes-modules %S/Inputs/APINotes  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -DFROM_FRAMEWORK=1 -verify

@import SomeOtherKit;

void test(A *a) {
#if FROM_FRAMEWORK
  [a methodA]; // expected-error{{unavailable}}
  [a methodB];

  // expected-note@SomeOtherKit/SomeOtherKit.h:5{{'methodA' has been explicitly marked unavailable here}}
#elif FROM_SEARCH_PATH
  [a methodA];
  [a methodB]; // expected-error{{unavailable}}

  // expected-note@SomeOtherKit/SomeOtherKit.h:6{{'methodB' has been explicitly marked unavailable here}}
#else
#  error Not something we need to test
#endif
}
