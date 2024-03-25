// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t -I %S/Inputs/shadowed-submodule/Foo -I %S/Inputs/shadowed-submodule/A2 %s -verify

@import Foo; // expected-error {{module 'A' was built in directory}}
             // expected-note@shadowed-submodule.m:4 {{imported by module 'Foo'}}
