// RUN: %clang_cc1 -std=c2y -fsyntax-only -verify -fblocks %s

void f1(id y) {
    l1: for (id x in y) {
        break l1;
        continue l1;
    }

    l2: for (id x in y) {
        break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
        continue l1; // expected-error {{'continue' label does not name an enclosing loop}}
    }

    l3: for (id x in y) {
        l4: for (id x in y) {
            break l3;
            break l4;
            continue l3;
            continue l4;
        }
    }
}

void f2(id y) {
    l1: for (id x in ({
        break l1; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
        continue l1; // expected-error {{'continue' label does not name an enclosing loop}}
        y;
    })) {}
}

void f3(id y) {
  a: b: for (id x in y) {
    (void) ^{
      break a; // expected-error {{'break' label does not name an enclosing loop or 'switch'}}
      continue b; // expected-error {{'continue' label does not name an enclosing loop}}
    };
  }
}
