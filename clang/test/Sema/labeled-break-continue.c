// RUN: %clang_cc1 -std=c2y -verify -fsyntax-only -fblocks %s
// RUN: %clang_cc1 -x c++ -verify -fsyntax-only -fblocks %s

void f1() {
  l1: while (true) {
    break l1;
    continue l1;
  }

  l2: for (;;) {
    break l2;
    continue l2;
  }

  l3: do {
    break l3;
    continue l3;
  } while (true);

  l4: switch (1) {
    case 1:
      break l4;
  }
}

void f2() {
  l1:;
  break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
  continue l1; // expected-error {{'continue' label does not name an enclosing loop}}

  l2: while (true) {
    break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue l1; // expected-error {{'continue' label does not name an enclosing loop}}
  }

  while (true) {
    break l2; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue l2; // expected-error {{'continue' label does not name an enclosing loop}}
  }

  break l3; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
  continue l3; // expected-error {{'continue' label does not name an enclosing loop}}
  l3: while (true) {}
}

void f3() {
  a: b: c: d: while (true) {
    break a;
    break b;
    break c;
    break d;

    continue a;
    continue b;
    continue c;
    continue d;

    e: while (true) {
      break a;
      break b;
      break c;
      break d;
      break e;

      continue a;
      continue b;
      continue c;
      continue d;
      continue e;
    }

    break e; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue e; // expected-error {{'continue' label does not name an enclosing loop}}
  }
}

void f4() {
  a: switch (1) {
    case 1: {
      continue a; // expected-error {{label of 'continue' refers to a switch statement}}
    }
  }
}

void f5() {
  a: {
    break a; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
  }

  b: {
    while (true)
      break b; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
  }
}

void f6() {
  a: while (({
    break a; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue a; // expected-error {{'continue' label does not name an enclosing loop}}
    1;
  })) {}

  b: for (
    int x = ({
      break b; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue b; // expected-error {{'continue' label does not name an enclosing loop}}
      1;
    });
    ({
      break b; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue b; // expected-error {{'continue' label does not name an enclosing loop}}
      1;
    });
    (void) ({
      break b; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue b; // expected-error {{'continue' label does not name an enclosing loop}}
      1;
    })
  ) {}

  c: do {} while (({
    break c; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue c; // expected-error {{'continue' label does not name an enclosing loop}}
    1;
  }));

  d: switch (({
    break d; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue d; // expected-error {{'continue' label does not name an enclosing loop}}
    1;
  })) { case 1:; }
}

void f7() {
  a: b: while (true) {
    (void) ^{
      break a; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue b; // expected-error {{'continue' label does not name an enclosing loop}}
    };
  }

  while (true) {
    break c; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
    continue d; // expected-error {{'continue' label does not name an enclosing loop}}
  }
}
