// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify 

import mod // expected-error {{'import' directive must end with a ';' on the same line}}
