// RUN: %clang_analyze_cc1 -analyzer-checker=core -std=c++26 -verify %s

void clang_analyzer_eval(bool);

template <class T>
constexpr decltype(auto) get0(const T& val) noexcept {
    auto& [...members] = val;
    auto&& r = members...[0]; // no-crash
    return r;
}

struct A {
    int a;
};

void no_crash_negative() {
    const int& x = get0(A{1});
    clang_analyzer_eval(x == 1);
}

void uninitialized() {
    A a;
    const int& x = get0(a);
    clang_analyzer_eval(x == 0); // expected-warning{{The left operand of '==' is a garbage value}}
}

template <int I, auto...Ts>
int index_template_pack()
{
  return Ts...[I]; // no-crash
}

void template_pack_no_crash()
{
  int r = index_template_pack<2, 0, 1, 42>();
}
