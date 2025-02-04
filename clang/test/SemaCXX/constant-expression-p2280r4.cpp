// RUN: %clang_cc1 -std=c++23 -verify %s
// RUN: %clang_cc1 -std=c++23 -verify %s -fexperimental-new-constant-interpreter

using size_t = decltype(sizeof(0));

namespace std {
struct type_info {
  const char* name() const noexcept(true);
};
}

template <typename T, size_t N>
constexpr size_t array_size(T (&)[N]) {
  return N;
}

void use_array(int const (&gold_medal_mel)[2]) {
  constexpr auto gold = array_size(gold_medal_mel); // ok
}

constexpr auto olympic_mile() {
  const int ledecky = 1500;
  return []{ return ledecky; };
}
static_assert(olympic_mile()() == 1500); // ok

struct Swim {
  constexpr int phelps() { return 28; }
  virtual constexpr int lochte() { return 12; }
  int coughlin = 12;
};

constexpr int how_many(Swim& swam) {
  Swim* p = &swam;
  return (p + 1 - 1)->phelps();
}

void splash(Swim& swam) {
  static_assert(swam.phelps() == 28);     // ok
  static_assert((&swam)->phelps() == 28); // ok
  Swim* pswam = &swam;                    // expected-note {{declared here}}
  static_assert(pswam->phelps() == 28);   // expected-error {{static assertion expression is not an integral constant expression}}
                                          // expected-note@-1 {{read of non-constexpr variable 'pswam' is not allowed in a constant expression}}
  static_assert(how_many(swam) == 28);    // ok
  static_assert(Swim().lochte() == 12);   // ok
  static_assert(swam.lochte() == 12);     // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(swam.coughlin == 12);     // expected-error {{static assertion expression is not an integral constant expression}}
}

extern Swim dc;
extern Swim& trident;

constexpr auto& sandeno   = typeid(dc);         // ok: can only be typeid(Swim)
constexpr auto& gallagher = typeid(trident);    // expected-error {{constexpr variable 'gallagher' must be initialized by a constant expression}}

namespace explicitThis {
struct C {
  constexpr int b()  { return 0; };

  constexpr int f(this C &c) {
    return c.b();     // ok
  }

   constexpr int g() {
    return f();       // ok
  }
};

void g() {
  C c;
  constexpr int x = c.f();
  constexpr int y = c.g();
}
}

namespace GH64376 {
template<int V>
struct Test {
    static constexpr int value = V;
};

int main() {
    Test<124> test;
    auto& test2 = test;

    if constexpr(test2.value > 3) {
       return 1;
    }

    return 0;
}
}

namespace GH30060 {
template<int V>
struct A {
  static constexpr int value = V;
};

template<class T>
static void test1(T &f) {
    A<f.value> bar;
}

void g() {
    A<42> f;

    test1(f);
}
}

namespace GH26067 {
struct A {
    constexpr operator int() const { return 42; }
};

template <int>
void f() {}

void test(const A& value) {
    f<value>();
}

int main() {
    A a{};
    test(a);
}
}

namespace GH34365 {
void g() {
  auto f = []() { return 42; };
  constexpr int x = f();
  [](auto f) { constexpr int x = f(); }(f);
  [](auto &f) { constexpr int x = f(); }(f);
  (void)[&]() { constexpr int x = f(); };
}
}

namespace GH118063 {
template <unsigned int N>
struct array {
    constexpr auto size() const -> unsigned int {
        return N;
    }
};

constexpr auto f(array<5> const& arr) {
    return array<arr.size()>{}.size();
}

int g() {
    array<5> arr {};
    static_assert(f(arr) == 5);
}
}
