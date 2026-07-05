// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -std=c++11 %s 2>&1 | FileCheck %s

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

namespace PR19176 {
struct Vector {
  struct iterator {
    int &operator*();
    iterator &operator++();
    iterator &operator++(int);
    bool operator==(const iterator &) const;
  };
  iterator begin();
  iterator end();
};

void f() {
  Vector v;
  int a[] = {1, 2, 3, 4};
  for (auto foo   =     a) // expected-error {{range-based 'for' statement uses ':', not '='}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:19-[[@LINE-1]]:20}:":"
    (void)foo;
  for (auto i
      =
      v) // expected-error@-1 {{range-based 'for' statement uses ':', not '='}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:8}:":"
    (void)i;
#define FORRANGE(v, a) for (DECLVARWITHINIT(v) a)  // expected-note {{expanded from macro}}
#define DECLAUTOVAR(v) auto v
#define DECLVARWITHINIT(v) DECLAUTOVAR(v) =  // expected-note {{expanded from macro}}
  FORRANGE(i, a) {  // expected-error {{range-based 'for' statement uses ':', not '='}}

  }
}
}

namespace GH63010 {
void foo(int n) {
    int a[] = {1, 2, 3, 4, 5};
    {
        for (auto x = n ? 1 : 2 : a); // expected-error {{expected ';' in 'for' statement specifier}} \
                                    // expected-error {{expected expression}}
        for (int i = 1; auto x = n ? 1 : 2 : a); // expected-error {{expected ';' in 'for' statement specifier}}
    }
    {
        for (auto x = n ? 1 : 2 : a)  // expected-error {{expected ';' in 'for' statement specifier}} \
                                      // expected-error {{expected expression}}

    }  // expected-error {{expected statement}}
    {
        for (int i = 1; auto x = n ? 1 : 2 : a) // expected-error {{expected ';' in 'for' statement specifier}}
    } // expected-error {{expected statement}}
}
}
