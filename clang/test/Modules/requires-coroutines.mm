// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify -fcoroutines-ts -DCOROUTINES


#ifdef COROUTINES
@import DependsOnModule.Coroutines;
@import DependsOnModule.NotCoroutines; // expected-error {{module 'DependsOnModule.NotCoroutines' is incompatible with feature 'coroutines'}}
#else
@import DependsOnModule.NotCoroutines;
@import DependsOnModule.Coroutines; // expected-error {{module 'DependsOnModule.Coroutines' requires feature 'coroutines'}}
#endif
