// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=pragma_pack_set %S/Inputs/module.map
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=pragma_pack_push %S/Inputs/module.map
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=pragma_pack_empty %S/Inputs/module.map
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodules-cache-path=%t -fmodule-name=pragma_pack_reset_push %S/Inputs/module.map
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fmodules -fimplicit-module-maps -x objective-c -verify -fmodules-cache-path=%t -I %S/Inputs %s
// FIXME: When we have a syntax for modules in C, use that.

@import pragma_pack_set;

#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack (pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}

@import pragma_pack_push;

#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 2}}
#pragma pack (pop)
#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack (pop)
#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack (pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}

#pragma pack (16)

@import pragma_pack_empty;

#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 16}}
#pragma pack (pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}

@import pragma_pack_reset_push;

#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack (pop)
#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 8}}

