// RUN: %clang_cc1 -fsyntax-only -verify %s

int x __attribute__((constructor)); // expected-warning {{'constructor' attribute only applies to functions}}
int f() __attribute__((constructor));
int f() __attribute__((constructor(1)));
int f() __attribute__((constructor(1,2))); // expected-error {{attribute takes no more than 1 argument}}
int f() __attribute__((constructor(1.0))); // expected-error {{'constructor' attribute requires parameter 1 to be an integer constant}}

int x __attribute__((destructor)); // expected-warning {{'destructor' attribute only applies to functions}}
int f() __attribute__((destructor));
int f() __attribute__((destructor(1)));
int f() __attribute__((destructor(1,2))); // expected-error {{attribute takes no more than 1 argument}}
int f() __attribute__((destructor(1.0))); // expected-error {{'destructor' attribute requires parameter 1 to be an integer constant}}


