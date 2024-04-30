// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -verify -x objective-c -fmodule-name=objcAtKeywordMissingEnd -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=Empty -emit-module %S/Inputs/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -verify -I %S/Inputs %s

@interface X // expected-note {{class started here}}
#pragma clang module import Empty // expected-error {{missing '@end'}}
