// RUN: %clang_cc1 -verify %s

3.2 // expected-error {{expected unqualified-id}}

extern "C" {
    typedef int Int;
}

Int foo(); // Ok
