// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wswitch-default %s

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

// Warn even completely covered Enum cases(GCC compatibility).
enum E { A, B };
enum E check_enum(enum E e) {
  switch (e) {                // expected-warning {{'switch' missing 'default' label}}
    case A: break;
    case B: break;
  }
  return e;
}

template<typename Index>
int t1(Index i)
{
  switch (i) {              // expected-warning {{'switch' missing 'default' label}}
    case 0: return 0;
    case 1: return 1;
  }
  return 0;
}

template<typename Index>
int t2(Index i)
{
  switch (i) {            // no-warning
    case 0: return 0;
    case 1: return 1;
    default: return 2;
  }
  return 0;
}

int main() {
  return t1(1);       // expected-note {{in instantiation of function template specialization 't1<int>' requested here}}
}

