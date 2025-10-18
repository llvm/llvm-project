// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/import_is_first_decl.cpp -fsyntax-only -verify

//--- import_is_first_decl.cpp
import std; // expected-error {{module import declaration can only appears in global module fragment, module interface or module implementation}}
// expected-note@-1 {{add 'module;' to the start of the file to introduce a global module fragment}}
// expected-error@-2 {{module 'std' not found}}
