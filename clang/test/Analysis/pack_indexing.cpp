// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++26 -verify %s

template <class T> void clang_analyzer_dump(T);

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
    clang_analyzer_dump(x); // expected-warning {{1 S32b}}
}

void uninitialized() {
    A a;
    const int& x = get0(a);
    clang_analyzer_dump(x); // expected-warning {{1st function call argument is an uninitialized value}}
}

void initialized() {
    A a;
    a.a = 4;
    const int& x = get0(a);
    clang_analyzer_dump(x); // expected-warning {{4 S32b}}
}

template <int I, auto...Ts>
int index_template_pack() {
  return Ts...[I]; // no-crash
}

void template_pack_no_crash() {
  (void)index_template_pack<2, 0, 1, 42>();
}
