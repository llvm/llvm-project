// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -I %S/Inputs/DependsOnModule.framework %s -verify

@import DependsOnModule.CXX;
// expected-error@module.map:11 {{module 'DependsOnModule.NotCXX' is incompatible with feature 'cplusplus'}}
@import DependsOnModule.NotCXX; // expected-note {{module imported here}}
// expected-error@module.map:15 {{module 'DependsOnModule.NotObjC' is incompatible with feature 'objc'}}
@import DependsOnModule.NotObjC; // expected-note {{module imported here}}
