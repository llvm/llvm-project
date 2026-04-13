// RUN: %clang_analyze_cc1 -std=c++20 -analyzer-checker=core -verify %s

// expected-no-diagnostics

template<typename... F>
struct overload : public F...
{
  using F::operator()...;
};

template<typename... F>
overload(F&&...) -> overload<F...>;

int main()
{
  const auto l = overload([](const int* i) {}); // no-crash

  return 0;
}
