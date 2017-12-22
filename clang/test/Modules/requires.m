// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -I %S/Inputs %s -verify -fmodule-feature custom_req1

// expected-error@DependsOnModule.framework/module.map:7 {{module 'DependsOnModule.CXX' requires feature 'cplusplus'}}
@import DependsOnModule.CXX; // expected-note {{module imported here}}
@import DependsOnModule.NotCXX;
// expected-error@DependsOnModule.framework/module.map:15 {{module 'DependsOnModule.NotObjC' is incompatible with feature 'objc'}}
@import DependsOnModule.NotObjC; // expected-note {{module imported here}}
@import DependsOnModule.CustomReq1; // OK
// expected-error@DependsOnModule.framework/module.map:22 {{module 'DependsOnModule.CustomReq2' requires feature 'custom_req2'}}
@import DependsOnModule.CustomReq2; // expected-note {{module imported here}}

@import RequiresWithMissingHeader; // OK
// expected-error@module.map:* {{module 'RequiresWithMissingHeader.HeaderBefore' requires feature 'missing'}}
@import RequiresWithMissingHeader.HeaderBefore; // expected-note {{module imported here}}
// expected-error@module.map:* {{module 'RequiresWithMissingHeader.HeaderAfter' requires feature 'missing'}}
@import RequiresWithMissingHeader.HeaderAfter; // expected-note {{module imported here}}
