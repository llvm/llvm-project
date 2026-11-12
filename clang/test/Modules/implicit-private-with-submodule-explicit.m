// Checks that the use of .Private to refer to _Private modules works with an
// explicit module.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -x objective-c -fmodules -fno-implicit-modules -emit-module -fmodule-name=A %t/module.modulemap -o %t/A.pcm
// RUN: %clang_cc1 -x objective-c -fmodules -fno-implicit-modules -emit-module -fmodule-name=A_Private %t/module.modulemap -o %t/A_Private.pcm

// Check lazily-loaded module
// RUN: %clang_cc1 -x objective-c -verify -fmodules -fno-implicit-modules -fmodule-file=A=%t/A.pcm -fmodule-file=A_Private=%t/A_Private.pcm -fsyntax-only %t/tu.m

// Check eagerly-loaded module
// RUN: %clang_cc1 -x objective-c -verify -fmodules -fno-implicit-modules -fmodule-file=%t/A.pcm -fmodule-file=%t/A_Private.pcm -fsyntax-only %t/tu.m

//--- module.modulemap
module A { header "a.h" }
module A_Private { header "priv.h" }

//--- a.h

//--- priv.h
void priv(void);

//--- tu.m
@import A.Private; // expected-warning{{no submodule named 'Private' in module 'A'; using top level 'A_Private'}}
// expected-note@*:* {{defined here}}

void tu(void) {
  priv();
}