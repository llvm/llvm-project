// RUN: rm -rf %t
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-canonical -emit-pch -o %t-A.pch %s -Wprivate-module

#ifndef HEADER
#define HEADER

@import A.Private; // expected-warning {{no submodule named 'Private' in module 'A'; using top level 'A_Private'}}
// expected-note@Inputs/implicit-private-canonical/A.framework/Modules/module.private.modulemap:1{{module defined here}}

const int *y = &APRIVATE;

#endif
