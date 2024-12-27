// RUN: %clang_cc1 -std=c++2c -verify -fsyntax-only %s

template<typename... T>
struct S {
    T...1; // expected-error{{expected member name or ';' after declaration specifiers}}
    T...[; // expected-error{{expected expression}} \
           // expected-error{{expected ']'}} \
           // expected-note {{to match this '['}} \
           // expected-warning{{declaration does not declare anything}}

    T...[1; // expected-error{{expected ']'}} \
            // expected-note {{to match this '['}} \
           // expected-warning{{declaration does not declare anything}}

    T...[]; // expected-error{{expected member name or ';' after declaration specifiers}}

    void f(auto... v) {
        decltype(v...[1]) a = v...[1];
        decltype(v...[1]) b = v...[]; // expected-error{{expected expression}}

        decltype(v...[1]) c = v...[ ;  // expected-error{{expected expression}}\
                                      // expected-error{{expected ']'}} \
                                      // expected-note {{to match this '['}}
    }
};


template <typename...>
struct typelist{};

template <typename... T>
requires requires(T...[0]) { {T...[0](0)}; }
struct SS : T...[1] {
    [[maybe_unused]] T...[1] base = {};
    using foo = T...[1];
    SS()
    : T...[1]()
    {}
    typelist<T...[0]> a;
    const T...[0] f(T...[0] && p) noexcept((T...[0])0) {
        T...[0] (*test)(const volatile T...[0]**);
        thread_local T...[0] d;
        [[maybe_unused]] T...[0] a = p;
        auto ptr = new T...[0](0);
        (*ptr).~T...[0]();
        return T...[0](0);
        typename T...[1]::foo b = 0;
        T...[1]::i = 0;
        return (T...[0])(a);
        new T...[0];
        [[maybe_unused]] auto l = []<T...[0]>(T...[0][1]) -> T...[0]{return{};};
        [[maybe_unused]] auto _ = l.template operator()<T...[0]{}>({0});
    }
    operator T...[0]() const{}
};

struct base {
    using foo = int;
    static inline int i = 42;
};

int main() {
    SS<int, base>().f(0);
}


namespace GH111460 {
template <typename... T>
requires( ); // expected-error {{expected expression}}
struct SS {
    void f( ) {
        (*p).~T...[](); // expected-error {{use of undeclared identifier 'p'}}
    }
};
}

namespace GH119072 {

template<typename... Ts>
void foo() {
  decltype(Ts...[0]::t) value;
}

}
