// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify 

// This import directive is ill-formed, it's missing an ';' after 
// module name, but we try to recovery from error and import the module.
import mod // expected-error {{import directive must end with a ';'}}
           // expected-error@-1 {{module 'mod' not found}}
