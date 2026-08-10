// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T> requires true
concept C = true; // expected-error{{concept cannot have associated constraints}}

// TODO: Add test for other kinds of associated constraints once we have them.
