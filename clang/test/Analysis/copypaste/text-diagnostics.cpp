// RUN: %clang_cc1 -analyze -analyzer-output=text -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

void log();

int max(int a, int b) { // expected-warning{{Duplicate code detected}} // expected-note{{Duplicate code detected}}
  log();
  if (a > b)
    return a;
  return b;
}

int maxClone(int a, int b) { // expected-note{{Similar code here}}
  log();
  if (a > b)
    return a;
  return b;
}
