// RUN: %clang_cc1 -analyze -std=gnu++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics


bool foo1(int x) {
  start:
  if (x != 3) {
    ++x;
    void *ptr = &&start;
    goto start;
  }
  end:
  return false;
}

// Targeting a different label with the address-of-label operator.
bool foo2(int x) {
  start:
  if (x != 3) {
    ++x;
    void *ptr = &&end;
    goto start;
  }
  end:
  return false;
}

// Different target label in goto
bool foo3(int x) {
  start:
  if (x != 3) {
    ++x;
    void *ptr = &&start;
    goto end;
  }
  end:
  return false;
}

// FIXME: Can't detect same algorithm as in foo1 but with different label names.
bool foo4(int x) {
  foo:
  if (x != 3) {
    ++x;
    void *ptr = &&foo;
    goto foo;
  }
  end:
  return false;
}
