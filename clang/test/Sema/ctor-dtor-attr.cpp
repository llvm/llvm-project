// RUN: %clang_cc1 -fsyntax-only -verify -Wno-strict-prototypes %s

template <int I> [[gnu::constructor(I)]] void ok_ctor();
template <int I> __attribute__((constructor(I))) void ok2_ctor();

template <int *I> [[gnu::constructor(I)]] void bad_ctor(); // expected-error {{'constructor' attribute requires an integer constant}}
template <int *I> __attribute__((constructor(I))) void bad2_ctor(); // expected-error {{'constructor' attribute requires an integer constant}}

template <int I> [[gnu::destructor(I)]] void ok_ctor();
template <int I> __attribute__((destructor(I))) void ok2_dtor();

template <int *I> [[gnu::destructor(I)]] void bad_dtor(); // expected-error {{'destructor' attribute requires an integer constant}}
template <int *I> __attribute__((destructor(I))) void bad2_dtor(); // expected-error {{'destructor' attribute requires an integer constant}}
