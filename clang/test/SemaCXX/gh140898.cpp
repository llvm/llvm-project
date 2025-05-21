// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
template <typename T>
void s() {
  switch (Unknown tr = 1) // expected-error {{unknown type name 'Unknown'}}
    tr;
}

void abc() {
  s<int>();
}
