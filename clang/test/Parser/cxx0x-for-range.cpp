// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T, typename U>
struct pair {};

template<typename T, typename U>
struct map {
  typedef pair<T,U> *iterator;
  iterator begin();
  iterator end();
};

template<typename T, typename U>
pair<T,U> &tie(T &, U &);

int foo(map<char*,int> &m) {
  char *p;
  int n;

  for (pair<char*,int> x : m) {
    (void)x;
  }

  for (tie(p, n) : m) { // expected-error {{for range declaration must declare a variable}}
    (void)p;
    (void)n;
  }

  return n;
}
