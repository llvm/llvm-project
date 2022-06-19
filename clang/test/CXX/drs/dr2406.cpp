// RUN: %clang_cc1 -x c++ %s  -verify

// dr2406: yes

void fallthrough(int n) {
  void g(), h(), i();
  switch (n) {
  case 1:
  case 2:
    g();
    [[fallthrough]];
  case 3: // warning on fallthrough discouraged
    do {
      [[fallthrough]]; // expected-error {{fallthrough annotation does not directly precede switch label}}
    } while (false);
  case 6:
    do {
      [[fallthrough]]; // expected-error {{fallthrough annotation does not directly precede switch label}}
    } while (n);
  case 7:
    while (false) {
      [[fallthrough]]; // expected-error {{fallthrough annotation does not directly precede switch label}}
    }
  case 5:
    h();
  case 4: // implementation may warn on fallthrough
    i();
    [[fallthrough]]; // expected-error {{fallthrough annotation does not directly precede switch label}}
  }
}
