// RUN: %clang_cc1 -std=c++2a -triple=x86_64-linux -fsyntax-only %s -verify

static_assert(true, "" // expected-warning {{'static_assert' with a user-generated message is a C++26 extension}} \
                       // expected-note {{to match this '('}}
                       // expected-error {{expected ')'}}
