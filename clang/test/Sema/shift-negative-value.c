// RUN: %clang_cc1 -x c -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -x c++ -std=c++98 -fsyntax-only -verify=cpp98 %s
// RUN: %clang_cc1 -x c++ -std=c++11 -fsyntax-only -verify -pedantic %s

enum shiftof {
    X = (-1<<29) // expected-warning {{folding it to a constant is a GNU extension}}
                 // expected-note@-1 {{left shift of negative value -1}}
                 // cpp98-error@-2 {{expression is not an integral constant expression}}
                 // cpp98-note@-3 {{left shift of negative value -1}}
};
