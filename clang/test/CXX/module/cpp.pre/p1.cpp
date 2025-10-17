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
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example1.cpp -D'DOT_BAR=.bar' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example2.cpp -D'MOD_ATTR=[[vendor::shiny_module]]' -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example3.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example4.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example5.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_example6.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_ext1.cpp -verify -E | FileCheck %t/cwg2947_ext1.cpp
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_ext2.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/cwg2947_ext3.cpp -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/preprocessed_module_file.cpp -E | FileCheck %t/preprocessed_module_file.cpp
// RUN: %clang_cc1 -std=c++20 %t/pedantic-errors.cpp -pedantic-errors -fsyntax-only -verify


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
module; int n;  // expected-warning {{extra tokens at end of 'module' directive}}
import foo; int n1; // expected-warning {{extra tokens at end of 'import' directive}}
                    // expected-error@-1 {{module 'foo' not found}}
const int *p1 = &n1;


//--- extra_tokens_after_module_decl2.cpp
export module m; int n2 // expected-warning {{extra tokens at end of 'module' directive}}
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
// expected-error@-1 {{module directive must end with a ';'}}

//--- semi_in_same_line.cpp
export module m // OK
[[]];

import foo // expected-error {{module 'foo' not found}}
;

//--- cwg2947_example1.cpp
export module foo DOT_BAR; // error: expansion of DOT_BAR; does not begin with ; or [
// expected-error@-1 {{module directive must end with a ';'}}

//--- cwg2947_example2.cpp
export module M MOD_ATTR ;        // OK
// expected-warning@-1 {{unknown attribute 'vendor::shiny_module' ignored}}

//--- cwg2947_example3.cpp
export module a
  .b;                         // error: preprocessing token after pp-module-name is not ; or [
// expected-error@-2 {{module directive must end with a ';'}}

//--- cwg2947_example4.cpp
export module M [[
  attr1,
  attr2 ]] ;                 // OK
// expected-warning@-2 {{unknown attribute 'attr1' ignored}}
// expected-warning@-2 {{unknown attribute 'attr2' ignored}}

//--- cwg2947_example5.cpp
export module M
  [[ attr1,
  attr2 ]] ;                 // OK
// expected-warning@-2 {{unknown attribute 'attr1' ignored}}
// expected-warning@-2 {{unknown attribute 'attr2' ignored}}

//--- cwg2947_example6.cpp
export module M; int // expected-warning {{extra tokens at end of 'module' directive}}
  n;                         // OK

//--- cwg2947_ext1.cpp
// CHECK: export __preprocessed_module m; int x;
// CHECK-NEXT: extern "C++" int *y = &x;
export module m; int x; // expected-warning {{extra tokens at end of 'module' directive}}
extern "C++" int *y = &x;

//--- cwg2947_ext2.cpp
export module x _Pragma("GCC warning \"Hi\""); // expected-warning {{Hi}}

//--- cwg2947_ext3.cpp
export module x; _Pragma("GCC warning \"hi\""); // expected-warning {{hi}}
// expected-warning@-1 {{extra tokens at end of 'module' directive}}

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
export module m; int n; // expected-warning {{extra tokens at end of 'module' directive}}
