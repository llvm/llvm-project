// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

template <typename T>
struct C {
    template <int N, typename U>
    friend void func(const C<U> &m) noexcept(N == 0);
};

template <int N, typename U>
void func(const C<U> &m) noexcept(N == 0) {}

int main() {
    C<int> t;
    return 0;
}
