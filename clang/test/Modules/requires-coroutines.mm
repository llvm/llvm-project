// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -Wno-private-module -F %S/Inputs -I %S/Inputs/DependsOnModule.framework %s -verify
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -Wno-private-module -F %S/Inputs -I %S/Inputs/DependsOnModule.framework %s -verify -fcoroutines-ts -DCOROUTINES

#ifdef COROUTINES
@import DependsOnModule.Coroutines;
// expected-error@module.map:29 {{module 'DependsOnModule.NotCoroutines' is incompatible with feature 'coroutines'}}
@import DependsOnModule.NotCoroutines; // expected-note {{module imported here}}
#else
@import DependsOnModule.NotCoroutines;
// expected-error@module.map:25 {{module 'DependsOnModule.Coroutines' requires feature 'coroutines'}}
@import DependsOnModule.Coroutines; // expected-note {{module imported here}}
#endif
