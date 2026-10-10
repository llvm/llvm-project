// RUN: %clang_cc1 -std=c++20 -verify=expected,no-new-interp -fsyntax-only -fnamed-loops %s -fno-experimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only -fnamed-loops %s -fexperimental-new-constant-interpreter

int a[10]{};
struct S {
  int a[10]{};
};

void f1() {
  l1: for (int x : a) {
    break l1;
    continue l1;
  }

  l2: for (int x : a) {
    break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue l1; // expected-error {{'continue' label does not name an enclosing loop}}
  }

  l3: for (int x : a) {
    l4: for (int x : a) {
      break l3;
      break l4;
      continue l3;
      continue l4;
    }
  }
}

void f2() {
  l1: for (
    int x = ({
      break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue l1; // expected-error {{'continue' label does not name an enclosing loop}}
      1;
    });
    int y : ({
      break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue l1; // expected-error {{'continue' label does not name an enclosing loop}}
      S();
    }).a
  ) {}
}

void f3() {
  a: while (true) {
    (void) []{
      break a; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue a; // expected-error {{'continue' label does not name an enclosing loop}}
    };
  }
}

void f4() {
  l1: for (;;) {
    constexpr int x = ({ // expected-error {{constexpr variable 'x' must be initialized by a constant expression}}
      break l1; // no-new-interp-note {{not supported in a constant expression}}
      1;
    });
  }
}

void f5() {
  l1: for (;;) {
    constexpr int x = ({ // expected-error {{constexpr variable 'x' must be initialized by a constant expression}}
      continue l1; // no-new-interp-note {{not supported in a constant expression}}
      1;
    });
  }
}

void f6() {
  l1: for (;;) {
    constexpr int x = ({ // expected-error {{constexpr variable 'x' must be initialized by a constant expression}}
      for (;;) break l1; // expected-note {{not supported in a constant expression}}
      1;
    });
  }
}

void f7() {
  l1: for (;;) {
    constexpr int x = ({ // expected-error {{constexpr variable 'x' must be initialized by a constant expression}}
      for (;;) continue l1; // expected-note {{not supported in a constant expression}}
      1;
    });
  }
}
