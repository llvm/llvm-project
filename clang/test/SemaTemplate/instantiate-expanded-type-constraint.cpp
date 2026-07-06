// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T, typename U>
concept same_as = is_same_v<T, U>; //#is_same_v

template<typename T, typename... Us>
concept either = (is_same_v<T, Us> || ...);

template<typename... Ts>
struct T {
    template<same_as<Ts>... Us>
    // expected-note@-1{{because 'same_as<int, bool>' evaluated to false}}
    // expected-note@#is_same_v{{because 'is_same_v<int, bool>' evaluated to false}}
    static void foo(Us... u, int x) { };
    // expected-note@-1{{candidate template ignored: deduced too few arguments}}
    // expected-note@-2{{candidate template ignored: constraints not satisfied}}

    template<typename... Us>
    struct S {
        template<either<Ts, Us...>... Vs>
        static void foo(Vs... v);
    };
};

int main() {
  T<int, bool>::foo(1); // expected-error{{no matching function for call to 'foo'}}
  T<int, bool>::foo(1, 2, 3); // expected-error{{no matching function for call to 'foo'}}
  T<int, bool>::S<char>::foo(1, 'a');
  T<int, bool>::S<char>::foo('a', true);
}
