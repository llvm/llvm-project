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
module                  // expected-error {{the module directive is ill-formed, module contextual keyword must be immediately followed on the same line by an identifier, or a ';' after being at the start of a line, or preceded by an export keyword at the start of a line}}
;export module M;       // expected-error {{the module directive is ill-formed, module contextual keyword must be immediately followed on the same line by an identifier, or a ';' after being at the start of a line, or preceded by an export keyword at the start of a line}}

//--- foo.cppm
export module foo;

//--- import_decl_not_in_same_line.cpp
export module M;
export                  // expected-error {{the import directive is ill-formed, import contextual keyword must be immediately followed on the same line by an identifier, '<', '"', or ':', but not '::', after being at the start of a line or preceded by an export at the start of the line}}
import
foo;

export                  // expected-error {{the import directive is ill-formed, import contextual keyword must be immediately followed on the same line by an identifier, '<', '"', or ':', but not '::', after being at the start of a line or preceded by an export at the start of the line}}
import foo;

//--- not_import.cpp
export module M;
import ::               // expected-error {{use of undeclared identifier 'import'}}
import ->               // expected-error {{cannot use arrow operator on a type}}
