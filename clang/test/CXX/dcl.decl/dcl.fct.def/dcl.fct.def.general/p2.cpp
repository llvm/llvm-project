// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Incomplete;                 // expected-note 2{{forward declaration of 'Incomplete'}}
Incomplete f(Incomplete) = delete; // well-formed
Incomplete g(Incomplete) {}        // expected-error{{incomplete result type 'Incomplete' in function definition}}\
// expected-error{{variable has incomplete type 'Incomplete'}}
