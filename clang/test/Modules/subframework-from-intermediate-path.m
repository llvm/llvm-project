// RUN: rm -rf %t
// RUN: %clang_cc1 -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -F %S/Inputs/DependsOnModule.framework/Frameworks %s -verify

@import DependsOnModule;
@import SubFramework; // expected-error{{module 'SubFramework' not found}}
