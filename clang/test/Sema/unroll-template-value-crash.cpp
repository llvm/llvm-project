// RUN: %clang_cc1 -x c++ -verify %s
// expected-no-diagnostics

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
