// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -x c++ -std=c++98 -fsyntax-only -verify=cxx98 -pedantic %s
// RUN: %clang_cc1 -x c++ -std=c++11 -fsyntax-only -verify -pedantic %s

enum shiftof {
    X = (1<<32) // expected-warning {{folding it to a constant is a GNU extension}}
                // expected-note@-1 {{shift count 32 >= width of type 'int'}}
                // cxx98-error@-2 {{expression is not an integral constant expression}}
                // cxx98-note@-3 {{shift count 32 >= width of type 'int'}}
};
