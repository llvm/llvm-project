// RUN: %clang_cc1 -fsyntax-only -verify %s

void a(int i) {
    switch(i) {
        case 1:
  // expected-error@+1 {{expected ']'}}
  [[fallthrough;;
        case 2:
        ;
    // expected-error@+1 {{expected statement}}
    };
}
