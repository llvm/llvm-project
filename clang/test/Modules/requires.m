// RUN: rm -rf %t
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -I %S/Inputs %s -verify -fmodule-feature custom_req1
// RUN: %clang_cc1 -Rmodule-include-translation -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -I %S/Inputs %s -verify -std=c89 -DTEST_C_FEATURES
#ifndef TEST_C_FEATURES
// expected-error@DependsOnModule.framework/Modules/module.modulemap:7 {{module 'DependsOnModule.CXX' requires feature 'cplusplus'}}
@import DependsOnModule.CXX; // expected-note {{module imported here}}
@import DependsOnModule.NotCXX;
// expected-error@DependsOnModule.framework/Modules/module.modulemap:15 {{module 'DependsOnModule.NotObjC' is incompatible with feature 'objc'}}
@import DependsOnModule.NotObjC; // expected-note {{module imported here}}
@import DependsOnModule.CustomReq1; // OK
// expected-error@DependsOnModule.framework/Modules/module.modulemap:22 {{module 'DependsOnModule.CustomReq2' requires feature 'custom_req2'}}
@import DependsOnModule.CustomReq2; // expected-note {{module imported here}}

@import RequiresWithMissingHeader; // OK
// expected-error@module.modulemap:* {{module 'RequiresWithMissingHeader.HeaderBefore' requires feature 'missing'}}
@import RequiresWithMissingHeader.HeaderBefore; // expected-note {{module imported here}}
// expected-error@module.modulemap:* {{module 'RequiresWithMissingHeader.HeaderAfter' requires feature 'missing'}}
@import RequiresWithMissingHeader.HeaderAfter; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:40 {{module 'DependsOnModule.CXX11' requires feature 'cplusplus11'}}
@import DependsOnModule.CXX11; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:43 {{module 'DependsOnModule.CXX14' requires feature 'cplusplus14'}}
@import DependsOnModule.CXX14; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:46 {{module 'DependsOnModule.CXX17' requires feature 'cplusplus17'}}
@import DependsOnModule.CXX17; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:49 {{module 'DependsOnModule.CXX20' requires feature 'cplusplus20'}}
@import DependsOnModule.CXX20; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:52 {{module 'DependsOnModule.CXX23' requires feature 'cplusplus23'}}
@import DependsOnModule.CXX23; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:55 {{module 'DependsOnModule.CXX26' requires feature 'cplusplus26'}}
@import DependsOnModule.CXX26; // expected-note {{module imported here}}
#else
// expected-error@DependsOnModule.framework/Modules/module.modulemap:58 {{module 'DependsOnModule.C99' requires feature 'c99'}}
@import DependsOnModule.C99; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:61 {{module 'DependsOnModule.C11' requires feature 'c11'}}
@import DependsOnModule.C11; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:64 {{module 'DependsOnModule.C17' requires feature 'c17'}}
@import DependsOnModule.C17; // expected-note {{module imported here}}
// expected-error@DependsOnModule.framework/Modules/module.modulemap:67 {{module 'DependsOnModule.C23' requires feature 'c23'}}
@import DependsOnModule.C23; // expected-note {{module imported here}}
#endif
