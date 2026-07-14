// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -Wexpansion-stmt-missing-braces -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -Wpedantic -verify

void f() {
  template for (int x : {1})
    template for (int y : {1}) // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      ; // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
  template for (int x : {1})
    if (x) // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      ;
  template for (int x : {1})
    switch (x) // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      ;
  template for (int x : {1})
    for (;;) // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      ;
  template for (int x : {1})
    while (x) // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      ;
  template for (int x : {1})
    do // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      ;
    while (x);
  template for (int x : {1})
    return; // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
  template for (int x : {1})
    [ // expected-warning {{ISO C++ requires a compound statement to be the body of an expansion statement}}
      []] {}
  template for (int x : {1})
    foo: {} // expected-error {{labels are not allowed in expansion statements}}
}
