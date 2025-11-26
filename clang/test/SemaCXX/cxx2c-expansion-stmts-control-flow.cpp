// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fblocks -verify

void g(int);

void label() {
  template for (auto x : {1, 2}) {
    invalid1:; // expected-error {{labels are not allowed in expansion statements}}
    invalid2:; // expected-error {{labels are not allowed in expansion statements}}
    goto invalid1; // expected-error {{use of undeclared label 'invalid1'}}
  }

  template for (auto x : {1, 2}) {
    (void) [] {
      template for (auto x : {1, 2}) {
        invalid3:; // expected-error {{labels are not allowed in expansion statements}}
      }
      ok:;
    };

    (void) ^{
      template for (auto x : {1, 2}) {
        invalid4:; // expected-error {{labels are not allowed in expansion statements}}
      }
      ok:;
    };

    struct X {
      void f() {
        ok:;
      }
    };
  }

  // GNU local labels are allowed.
  template for (auto x : {1, 2}) {
    __label__ a;
    if (x == 1) goto a;
    a:;
    if (x == 1) goto a;
  }

  // Likewise, jumping *out* of an expansion statement is fine.
  template for (auto x : {1, 2}) {
    if (x == 1) goto lbl;
    g(x);
  }
  lbl:;
  template for (auto x : {1, 2}) {
    if (x == 1) goto lbl;
    g(x);
  }

  // Jumping into one is not possible, as local labels aren't visible
  // outside the block that declares them, and non-local labels are invalid.
  goto exp1; // expected-error {{use of undeclared label 'exp1'}}
  goto exp3; // expected-error {{use of undeclared label 'exp3'}}
  template for (auto x : {1, 2}) {
    __label__ exp1, exp2;
    exp1:;
    exp2:;
    exp3:; // expected-error {{labels are not allowed in expansion statements}}
  }
  goto exp2; // expected-error {{use of undeclared label 'exp2'}}

  // Allow jumping from inside an expansion statement to a local label in
  // one of its parents.
  out1:;
  template for (auto x : {1, 2}) {
    __label__ x, y;
    x:
    goto out1;
    goto out2;
    template for (auto x : {3, 4}) {
      goto x;
      goto y;
      goto out1;
      goto out2;
    }
    y:
  }
  out2:;
}


void case_default(int i) {
  switch (i) { // expected-note 3 {{switch statement is here}}
    template for (auto x : {1, 2}) {
      case 1:; // expected-error {{'case' belongs to 'switch' outside enclosing expansion statement}}
        template for (auto x : {1, 2}) {
          case 2:; // expected-error {{'case' belongs to 'switch' outside enclosing expansion statement}}
        }
      default: // expected-error {{'default' belongs to 'switch' outside enclosing expansion statement}}
        switch (i) {  // expected-note {{switch statement is here}}
          case 3:;
          default:
            template for (auto x : {1, 2}) {
              case 4:; // expected-error {{'case' belongs to 'switch' outside enclosing expansion statement}}
            }
        }
    }
  }

  template for (auto x : {1, 2}) {
    switch (i) {
      case 1:;
      default:
    }
  }

  // Ensure that we diagnose this even if the statements would be discarded.
  switch (i) { // expected-note 2 {{switch statement is here}}
    template for (auto x : {}) {
      case 1:; // expected-error {{'case' belongs to 'switch' outside enclosing expansion statement}}
      default:; // expected-error {{'default' belongs to 'switch' outside enclosing expansion statement}}
    }
  }
}
