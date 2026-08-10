// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

[[clang::external_source_symbol(language="Swift", defined_in="module")]]
void function() { }
