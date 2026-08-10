// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

id x = @"foo"_bar; // expected-error{{user-defined suffix cannot be used here}}
