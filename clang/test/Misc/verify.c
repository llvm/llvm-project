// RUN: %clang_cc1 -fsyntax-only -verify %s

struct s; // expected-note 7 {{forward declaration of 'struct s'}}

// standard string matching
struct s s1; // expected-error {{tentative definition has type 'struct s' that is never completed}}
struct s s2; // expected-error {{tentative definition has type}}

// regex matching
struct s r1; // expected-error    {{tentative definition has type 'struct s' that is never completed}}
struct s r2; // expected-error-re {{tentative definition has type '{{.*[[:space:]]*.*}}' that is never completed}}
struct s r3; // expected-error-re {{tentative definition has type '{{(.*)[[:space:]]*(.*)}}' that is never completed}}
struct s r4; // expected-error-re {{{{^}}tentative}}
struct s r5; // expected-error-re {{completed{{$}}}}
