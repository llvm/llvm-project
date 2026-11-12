// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A = B; // expected-error{{namespace name}}

namespace A = !; // expected-error {{expected namespace name}}
namespace A = A::!; // expected-error {{expected namespace name}} \
                    // expected-error{{use of undeclared identifier 'A'}}


