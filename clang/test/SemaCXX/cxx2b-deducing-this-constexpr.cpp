// RUN: %clang_cc1 -fsyntax-only -std=c++2b %s -verify
// RUN: %clang_cc1 -fsyntax-only -std=c++2b %s -verify -fexperimental-new-constant-interpreter
// expected-no-diagnostics

template <typename Base>
struct Wrap : Base {

};

struct S {
    constexpr int f(this const S&) {
        return 42;
    }
    constexpr int f(this const S&, auto&&... args) {
        return (args + ... + 0);
    }
    constexpr int operator[](this const S&) {
        return 42;
    }
    constexpr int operator[](this const S& self, int i) {
        return i + self.base;
    }
    constexpr int operator()(this const S&) {
        return 42;
    }
    constexpr int operator()(this const S& self, int i) {
        return self.base + i;
    }
    constexpr bool operator==(this const S& self, auto && test) {
        return self.base == test;
    };
    constexpr int operator*(this const S& self) {
        return self.base + 22;
    };
    constexpr operator Wrap<S> (this const S& self) {
        return Wrap<S>{self};
    };
    constexpr int operator <<(this Wrap<S> self, int i) {
        return self.base+i;
    }

    int base = 20;
};

consteval void test() {
    constexpr S s;
    static_assert(s.f() == 42);
    static_assert(s[] == 42);
    static_assert(s[22] == 42);
    static_assert(s.f() == 42);
    static_assert(s() == 42);
    static_assert(s(22) == 42);
    static_assert(s == 20);
    static_assert(s != 0);
    static_assert(*s == 42);
    static_assert((s << 11) == 31);
}

namespace GH68070 {

constexpr auto f = [x = 3]<typename Self>(this Self&& self) {
    return x;
};

auto g = [x = 3]<typename Self>(this Self&& self) {
    return x;
};

int test() {
  constexpr int a = f();
  static_assert(a == 3);
  return f() + g();
}

}
