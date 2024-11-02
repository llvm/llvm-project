// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-note@1 15{{add 'module;' to the start of the file to introduce a global module fragment}}

module std;    // expected-error {{'std' is a reserved name for a module}}
module _Test;  // expected-error {{'_Test' is a reserved name for a module}} \
                  expected-error {{module declaration must occur at the start of the translation unit}}
module module; // expected-error {{'module' is an invalid name for a module}} \
                  expected-error {{module declaration must occur at the start of the translation unit}}
module std0;   // expected-error {{'std0' is a reserved name for a module}} \
                  expected-error {{module declaration must occur at the start of the translation unit}}

export module module; // expected-error {{'module' is an invalid name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module import; // expected-error {{'import' is an invalid name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module _Test;  // expected-error {{'_Test' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module __test; // expected-error {{'__test' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module te__st; // expected-error {{'te__st' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module std;    // expected-error {{'std' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module std.foo;// expected-error {{'std' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module std0;   // expected-error {{'std0' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module std1000000; // expected-error {{'std1000000' is a reserved name for a module}} \
                         expected-error {{module declaration must occur at the start of the translation unit}}
export module should_fail._Test; // expected-error {{'_Test' is a reserved name for a module}} \
                                    expected-error {{module declaration must occur at the start of the translation unit}}

// Show that being in a system header doesn't save you from diagnostics about
// use of an invalid module-name identifier.
# 34 "reserved-names-1.cpp" 1 3
export module module;       // expected-error {{'module' is an invalid name for a module}} \
                               expected-error {{module declaration must occur at the start of the translation unit}}

export module _Test.import; // expected-error {{'import' is an invalid name for a module}} \
                               expected-error {{module declaration must occur at the start of the translation unit}}
# 39 "reserved-names-1.cpp" 2 3

// We can still use a reserved name on imoport.
import std; // expected-error {{module 'std' not found}}
