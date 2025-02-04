// RUN: %clang_cc1 -fsyntax-only -verify=expected,c -pedantic %s
// RUN: %clang_cc1 -x c++ -std=c++98 -fsyntax-only -verify=expected,cpp %s
// RUN: %clang_cc1 -x c++ -std=c++11 -fsyntax-only -verify=expected,cpp %s

enum shiftof {
    X = (1<<32) // c-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                // cpp-error@-1 {{expression is not an integral constant expression}}
                // expected-note@-2 {{shift count 32 >= width of type 'int'}}
};
