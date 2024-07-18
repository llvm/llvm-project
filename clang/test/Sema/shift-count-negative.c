// RUN: %clang_cc1 -x c -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -pedantic %s

// RUN: %clang_cc1 -x c -fsyntax-only -verify -pedantic %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -pedantic %s -fexperimental-new-constant-interpreter

// cpp-no-diagnostics

enum shiftof {
    X = (1<<-29) // expected-warning {{folding it to a constant is a GNU extension}}
                 // expected-note@-1 {{negative shift count -29}}
};
