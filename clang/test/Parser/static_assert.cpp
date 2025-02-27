// RUN: %clang_cc1 -fsyntax-only -triple=x86_64-linux -std=c++2a -verify=cxx2a %s
// RUN: %clang_cc1 -fsyntax-only -triple=x86_64-linux -std=c++2c -verify=cxx2c %s

static_assert(true, "" // cxx2a-warning {{'static_assert' with a user-generated message is a C++26 extension}} \
                       // cxx2a-note {{to match this '('}} cxx2c-note {{to match this '('}}
                       // cxx2a-error {{expected ')'}}     cxx2c-error {{expected ')'}}
