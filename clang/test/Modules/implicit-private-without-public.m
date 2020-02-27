// REQUIRES: shell
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -F%S/Inputs/implicit-private-without-public \
// RUN:   -fsyntax-only %s -verify

@import Foo_Private;

// Private module map without a public one isn't supported for deprecated module map locations.
@import DeprecatedModuleMapLocation_Private;
// expected-error@-1{{module 'DeprecatedModuleMapLocation_Private' not found}}
