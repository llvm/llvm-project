// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-keyword %s

// Show that @class is not diagnosed as use of a C++ keyword in Objective-C,
// but that other uses of class are diagnosed.

@class Foo; // Okay, Objective-C @ keyword, not a regular identifier

int class = 12; // expected-warning {{identifier 'class' conflicts with a C++ keyword}}

