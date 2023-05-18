// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic
// PR7072
()( // expected-error {{expected unqualified-id}}

