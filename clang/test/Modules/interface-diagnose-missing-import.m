// RUN: rm -rf %t
// RUN: %clang_cc1 %s -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F%S/Inputs/interface-diagnose-missing-import -verify
// expected-no-diagnostics
@interface Buggy
@end

@import Foo.Bar;

// No diagnostic for inaccessible 'Buggy' definition because we have another definition right in this file.
@interface Buggy (MyExt)
@end
