// RUN: %clang_cc1 -fsyntax-only -verify %s

int x __attribute__((constructor)); // expected-warning {{'constructor' attribute only applies to functions}}
int f(void) __attribute__((constructor));
int f(void) __attribute__((constructor(1)));
int f(void) __attribute__((constructor(1,2))); // expected-error {{'constructor' attribute takes no more than 1 argument}}
int f(void) __attribute__((constructor(1.0))); // expected-error {{'constructor' attribute requires an integer constant}}
int f(void) __attribute__((constructor(0x100000000))); // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}

int x __attribute__((destructor)); // expected-warning {{'destructor' attribute only applies to functions}}
int f(void) __attribute__((destructor));
int f(void) __attribute__((destructor(1)));
int f(void) __attribute__((destructor(1,2))); // expected-error {{'destructor' attribute takes no more than 1 argument}}
int f(void) __attribute__((destructor(1.0))); // expected-error {{'destructor' attribute requires an integer constant}}

void knr() __attribute__((constructor));
