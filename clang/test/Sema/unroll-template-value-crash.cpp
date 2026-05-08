// RUN: %clang_cc1 -x c++ -verify %s

template <int Unroll> void foo() {
  #pragma unroll Unroll
  for (int i = 0; i < Unroll; ++i);

  #pragma GCC unroll Unroll
  for (int i = 0; i < Unroll; ++i);
}

struct val {
  constexpr operator int() const { return 1; }
};

// generic lambda (using double template instantiation)

template<typename T>
void use(T t) {
  auto lam = [](auto N) {
    #pragma clang loop unroll_count(N+1)
    for (int i = 0; i < 10; ++i);

    #pragma unroll N+1
    for (int i = 0; i < 10; ++i);

    #pragma GCC unroll N+1
    for (int i = 0; i < 10; ++i);
  };
  lam(t);
}

void test() {
  use(val());
}

template <typename T> void pr49502(T v) {
#pragma GCC unroll v() // expected-error {{called object type 'int' is not a function or function pointer}}
  for (;;) {
  }
}

void pr49502_caller() {
  pr49502(0); // expected-note {{in instantiation of function template specialization 'pr49502<int>' requested here}}
}

template <typename T, typename U> struct pr49502_partial;

template <typename T> struct pr49502_partial<T, T> {
  static void dependent(T v) {
#pragma GCC unroll v() // expected-error {{called object type 'int' is not a function or function pointer}}
    for (;;) {
    }
  }

  static void mixed(T v) {
#pragma GCC unroll v() + 1 // expected-error {{called object type 'int' is not a function or function pointer}}
    for (;;) {
    }
  }

  static void non_dependent(int v) {
#pragma GCC unroll v() // expected-error {{called object type 'int' is not a function or function pointer}}
    for (;;) {
    }
  }
};

void pr49502_partial_caller() {
  pr49502_partial<int, int>::dependent(0); // expected-note {{in instantiation of member function 'pr49502_partial<int, int>::dependent' requested here}}
  pr49502_partial<int, int>::mixed(0); // expected-note {{in instantiation of member function 'pr49502_partial<int, int>::mixed' requested here}}
}
