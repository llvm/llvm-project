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
