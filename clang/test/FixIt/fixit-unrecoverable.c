/* FIXME: This is a file containing various typos for which we can
   suggest corrections but are unable to actually recover from
   them. Ideally, we would eliminate all such cases and move these
   tests elsewhere. */

// RUN: %clang_cc1 -fsyntax-only -verify %s

unsinged x = 17; // expected-error{{unknown type name 'unsinged'; did you mean 'unsigned'?}}
