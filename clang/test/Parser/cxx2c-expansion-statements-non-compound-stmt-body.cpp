// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -Wexpansion-stmt-missing-braces -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -Wpedantic -verify

void f() {
  template for (int x : {1})
    template for (int y : {1}) // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      ; // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
  template for (int x : {1})
    if (x) // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      ;
  template for (int x : {1})
    switch (x) // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      ;
  template for (int x : {1})
    for (;;) // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      ;
  template for (int x : {1})
    while (x) // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      ;
  template for (int x : {1})
    do // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      ;
    while (x);
  template for (int x : {1})
    return; // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
  template for (int x : {1})
    [] {}(); // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
  template for (int x : {1})
    [ // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      []] if (x)
      ;
  template for (int x : {1})
    [ // expected-warning {{ISO C++ requires the body of an expansion statement to be a compound statement}}
      [likely]] if (x)
      ;
  template for (int x : {1})
    [ // expected-warning {{ISO C++ forbids attributes before the compound statement of an expansion statement}}
      []] {}
  template for (int x : {1})
    [ // expected-warning {{ISO C++ forbids attributes before the compound statement of an expansion statement}}
      [likely]] {}
  template for (int x : {1})
    __attribute__ // expected-warning {{ISO C++ forbids attributes before the compound statement of an expansion statement}}
      (()) {}
  template for (int x : {1})
    foo: {} // expected-error {{labels are not allowed in expansion statements}}
}
