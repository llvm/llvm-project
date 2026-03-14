// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wmain -verify %s

// expected-note@+1 {{previous definition is here}}
int main() {
  return 0;
}  // no-warning

// expected-error@+1 {{redefinition of 'main'}}
int main() {
  return 1.0;
}  // no-warning

int main() {
  bool b = true;
  return b;  // no-warning
}

int main() {
  return true;  // expected-warning {{bool literal returned from 'main'}}
}
