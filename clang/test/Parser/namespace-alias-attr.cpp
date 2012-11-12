// RUN: %clang_cc1 -verify %s

namespace A
{
}

namespace B __attribute__ (( static )) = A; // expected-error{{attributes can not be specified on namespace alias}}

