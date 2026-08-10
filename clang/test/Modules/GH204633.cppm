// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++26 -triple x86_64-pc-win32 -fsyntax-only -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file=%t/original.cppm -verify %t/original.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fmodules -verify %t/inline.cppm

// Use the same file as both the source input and the module map. The module map
// parser accepts this as a Clang module definition, while the C++ parser
// diagnoses it as a malformed module declaration.
//--- original.cppm
// expected-error@+2 {{unexpected preprocessing token '{' after module name}}
// expected-error@+1 {{module directive must end with a ';'}}
module M {}

// Build the conflicting Clang module inline.
//--- inline.cppm
// expected-no-diagnostics
#pragma clang module build Foo
module Foo {}
#pragma clang module endbuild

module Foo;
