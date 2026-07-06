// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/hash.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/module.cpp -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/rightpad.cppm -emit-module-interface -o %t/rightpad.pcm
// RUN: %clang_cc1 -std=c++20 %t/M_part.cppm -emit-module-interface -o %t/M_part.pcm
// RUN: %clang_cc1 -std=c++20 -xc++-system-header %t/string -emit-header-unit -o %t/string.pcm
// RUN: %clang_cc1 -std=c++20 -xc++-user-header %t/squee -emit-header-unit -o %t/squee.pcm
// RUN: %clang_cc1 -std=c++20 %t/import.cpp -isystem %t \
// RUN:                                     -fmodule-file=rightpad=%t/rightpad.pcm \
// RUN:                                     -fmodule-file=M:part=%t/M_part.pcm \
// RUN:                                     -fmodule-file=%t/string.pcm \
// RUN:                                     -fmodule-file=%t/squee.pcm \
// RUN:                                     -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/module_decl_not_in_same_line.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 %t/import_decl_not_in_same_line.cpp -fmodule-file=foo=%t/foo.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/not_import.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/import_spaceship.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/leading_empty_macro.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/operator_keyword_and.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/operator_keyword_and2.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/macro_in_module_decl_suffix.cpp -D'ATTR(X)=[[X]]' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/macro_in_module_decl_suffix2.cpp -D'ATTR(X)=[[X]]' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/extra_tokens_after_module_decl1.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/extra_tokens_after_module_decl2.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/object_like_macro_in_module_name.cpp -Dm=x -Dn=y -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/object_like_macro_in_partition_name.cpp -Dm=x -Dn=y -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/unexpected_character_in_pp_module_suffix.cpp -D'm(x)=x' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/semi_in_same_line.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/preprocessed_module_file.cpp -E | FileCheck %t/preprocessed_module_file.cpp
// RUN: %clang_cc1 -std=c++20 %t/pedantic-errors.cpp -pedantic-errors -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/xcpp-output.cpp -fsyntax-only -verify -xc++-cpp-output
// RUN: %clang_cc1 -std=c++20 %t/func_like_macro.cpp -D'm(x)=x' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/lparen.cpp -D'm(x)=x' -D'LPAREN=(' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/control_line.cpp -fsyntax-only -verify


//--- hash.cpp
// expected-no-diagnostics
#                       // preprocessing directive

//--- module.cpp
// expected-no-diagnostics
module ;                // preprocessing directive
export module leftpad;  // preprocessing directive

//--- string
#ifndef STRING_H
#define STRING_H
#endif // STRING_H

//--- squee
#ifndef SQUEE_H
#define SQUEE_H
#endif

//--- rightpad.cppm
export module rightpad;

//--- M_part.cppm
export module M:part;

//--- import.cpp
export module M;
import <string>;        // expected-warning {{the implementation of header units is in an experimental phase}}
export import "squee";  // expected-warning {{the implementation of header units is in an experimental phase}}
import rightpad;        // preprocessing directive
import :part;           // preprocessing directive

//--- module_decl_not_in_same_line.cpp
module                  // expected-error {{a type specifier is required for all declarations}}
;export module M;       // expected-error {{export declaration can only be used within a module interface}} \
                        // expected-error {{unknown type name 'module'}}

//--- foo.cppm
export module foo;

//--- import_decl_not_in_same_line.cpp
export module M;
export
import                  // expected-error {{unknown type name 'import'}}
foo;

export
import foo;             // expected-error {{unknown type name 'import'}}

//--- not_import.cpp
export module M;
import ::               // expected-error {{use of undeclared identifier 'import'}}
import ->               // expected-error {{cannot use arrow operator on a type}}

//--- import_spaceship.cpp
export module M;
import <=>; // expected-error {{'=' file not found}}

//--- leading_empty_macro.cpp
// expected-no-diagnostics
export module M;
typedef int import;
#define EMP
EMP import m; // The phase 7 grammar should see import as a typedef-name.

//--- operator_keyword_and.cpp
// expected-no-diagnostics
typedef int import;
extern
import and x;

//--- operator_keyword_and2.cpp
// expected-no-diagnostics
typedef int module;
extern
module and x;

//--- macro_in_module_decl_suffix.cpp
export module m ATTR(x);    // expected-warning {{unknown attribute 'x' ignored}}

//--- macro_in_module_decl_suffix2.cpp
export module m [[y]] ATTR(x);          // expected-warning {{unknown attribute 'y' ignored}} \
                                        // expected-warning {{unknown attribute 'x' ignored}}

//--- extra_tokens_after_module_decl1.cpp
module; int n;  // expected-warning {{extra tokens after semicolon in 'module' directive}}
import foo; int n1; // expected-warning {{extra tokens after semicolon in 'import' directive}}
                    // expected-error@-1 {{module 'foo' not found}}
const int *p1 = &n1;


//--- extra_tokens_after_module_decl2.cpp
export module m; int n2 // expected-warning {{extra tokens after semicolon in 'module' directive}}
;
const int *p2 = &n2;


//--- object_like_macro_in_module_name.cpp
export module m.n;
// expected-error@-1 {{module name component 'm' cannot be a object-like macro}}
// expected-note@* {{macro 'm' defined here}}
// expected-error@-3 {{module name component 'n' cannot be a object-like macro}}
// expected-note@* {{macro 'n' defined here}}

//--- object_like_macro_in_partition_name.cpp
export module m:n;
// expected-error@-1 {{module name component 'm' cannot be a object-like macro}}
// expected-note@* {{macro 'm' defined here}}
// expected-error@-3 {{partition name component 'n' cannot be a object-like macro}}
// expected-note@* {{macro 'n' defined here}}

//--- unexpected_character_in_pp_module_suffix.cpp
export module m();
// expected-error@-1 {{unexpected preprocessing token '(' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}

//--- semi_in_same_line.cpp
export module m // OK
[[]];

import foo // expected-error {{module 'foo' not found}}
;

//--- preprocessed_module_file.cpp
// CHECK: __preprocessed_module;
// CHECK-NEXT: export __preprocessed_module M;
// CHECK-NEXT: __preprocessed_import std;
// CHECK-NEXT: export __preprocessed_import bar;
// CHECK-NEXT: struct import {};
// CHECK-EMPTY:
// CHECK-NEXT: import foo;
module;
export module M;
import std;
export import bar;
struct import {};
#define EMPTY
EMPTY import foo;

//--- pedantic-errors.cpp
export module m; int n; // expected-warning {{extra tokens after semicolon in 'module' directive}}

//--- xcpp-output.cpp
// expected-no-diagnostics
typedef int module;
module x;

//--- func_like_macro.cpp
// #define m(x) x
export module m
     (foo); // expected-error {{unexpected preprocessing token '(' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}

//--- lparen.cpp
// #define m(x) x
// #define LPAREN (
export module m
    LPAREN foo); // expected-error {{unexpected preprocessing token 'LPAREN' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}

//--- control_line.cpp
#if 0 // #1
export module m; // expected-error {{module directive lines are not allowed on lines controlled by preprocessor conditionals}}
#else
export module m; // expected-error {{module directive lines are not allowed on lines controlled by preprocessor conditionals}} \
                 // expected-error {{module declaration must occur at the start of the translation unit}} \
                 // expected-note@#1 {{add 'module;'}}
#endif
