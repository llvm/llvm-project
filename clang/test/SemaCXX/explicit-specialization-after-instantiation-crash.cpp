// RUN: %clang_cc1 -fsyntax-only -verify %s
//
// Test that explicit template specialization after instantiation
// is handled gracefully without assertion failure.

template <typename T>
struct X {
    struct Y {
        Y() : v(0) {}
        int v;
        int getValue();
    } y;
};

template <typename T>
int X<T>::Y::getValue() {
    return ++v;
}

// expected-error@+1 {{explicit specialization of 'Y' after instantiation}}
template <> struct X<int>::Y { int getValue() { return 55; } };
// expected-note@-10 {{implicit instantiation first required here}}

extern template class X<int>::Y;

int main() {
    X<int> x;
    return x.y.getValue(); // expected-error {{no member named 'getValue' in 'X<int>::Y'}}
}
