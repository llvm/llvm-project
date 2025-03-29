// RUN: %clang_cc1 -x c -fsyntax-only -verify=expected,c -pedantic %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cpp %s

// RUN: %clang_cc1 -x c -fsyntax-only -verify=expected,c -pedantic %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=expected,cpp %s -fexperimental-new-constant-interpreter

enum shiftof {
    X = (1<<-29) // c-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                 // cpp-error@-1 {{expression is not an integral constant expression}}
                 // expected-note@-2 {{negative shift count -29}}
};
