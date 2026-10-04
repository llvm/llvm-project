// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s

// Check argument validation when the attribute is attached to a pointer type.
// A missing argument must be diagnosed without crashing (regression for
// https://github.com/llvm/llvm-project/issues/227549).
int * __attribute__((__swift_attr__)) missing; // expected-error {{'__swift_attr__' attribute takes one argument}}
int * __attribute__((swift_attr())) empty; // expected-error {{'swift_attr' attribute takes one argument}}
int * __attribute__((swift_attr("@A", "@B"))) extra; // expected-error {{'swift_attr' attribute takes one argument}}
int * __attribute__((swift_attr(1))) non_string; // expected-error {{expected string literal as argument of 'swift_attr' attribute}}
int * __attribute__((swift_attr("@A"))) valid;

// Declaration attributes should continue to use the common argument checks.
__attribute__((swift_attr)) int decl_missing; // expected-error {{'swift_attr' attribute takes one argument}}
__attribute__((swift_attr("@A", "@B"))) int decl_extra; // expected-error {{'swift_attr' attribute takes one argument}}
__attribute__((swift_attr("@A"))) int decl_valid;
