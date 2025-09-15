// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify 

import mod // expected-error {{expected ';' after module name}}
           // expected-error@-1 {{module 'mod' not found}}
