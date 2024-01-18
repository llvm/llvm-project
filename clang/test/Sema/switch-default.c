// RUN: %clang_cc1 -fsyntax-only -verify -Wswitch-default %s

int f1(int a) {
  switch (a) {                // expected-warning {{'switch' missing 'default' label}}
    case 1: a++; break;
    case 2: a += 2; break;
  }
  return a;
}

int f2(int a) {
  switch (a) {                // no-warning
    default:
      ;
  }
  return a;
}
