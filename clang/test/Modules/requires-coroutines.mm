// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -Wno-private-module -F %S/Inputs -I %S/Inputs/DependsOnModule.framework %s -verify

@import DependsOnModule.NotCoroutines;
// expected-error@Modules/module.modulemap:25 {{module 'DependsOnModule.Coroutines' requires feature 'coroutines'}}
@import DependsOnModule.Coroutines; // expected-note {{module imported here}}
