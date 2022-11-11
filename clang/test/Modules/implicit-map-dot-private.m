// RUN: rm -rf %t

// Implicit modules.
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -F %S/Inputs/implicit-private-canonical -fsyntax-only %s -Wprivate-module

// Explicit modules.
// RUN: %clang_cc1 -x objective-c -fmodules -emit-module -fmodule-name=A -o %t/A.pcm \
// RUN:   %S/Inputs/implicit-private-canonical/A.framework/Modules/module.modulemap -Wprivate-module
// RUN: %clang_cc1 -x objective-c -fmodules -emit-module -fmodule-name=A_Private -o %t/A_Private.pcm \
// RUN:   %S/Inputs/implicit-private-canonical/A.framework/Modules/module.private.modulemap -Wprivate-module
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fno-implicit-modules \
// RUN:   -fmodule-file=A=%t/A.pcm -fmodule-file=A_Private=%t/A_Private.pcm \
// RUN:   -F %S/Inputs/implicit-private-canonical -fsyntax-only %s -Wprivate-module

#ifndef HEADER
#define HEADER

@import A.Private; // expected-warning {{no submodule named 'Private' in module 'A'; using top level 'A_Private'}}
// expected-note@Inputs/implicit-private-canonical/A.framework/Modules/module.private.modulemap:1{{module defined here}}

const int *y = &APRIVATE;

#endif
