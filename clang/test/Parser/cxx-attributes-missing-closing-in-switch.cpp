// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-note@+1 {{to match this '{'}}
void a(int i) {
    switch(i) {
        case 1:
  // expected-error@+2 {{expected ']'}}
  // expected-error@+3 {{expected ']'}}
  [[fallthrough;;
        case 2:
        ;
    };
   // expected-error@+2 {{expected statement}}
   // expected-error@+1 {{expected '}'}}
}
