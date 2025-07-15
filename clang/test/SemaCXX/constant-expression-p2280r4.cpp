// RUN: %clang_cc1 -std=c++23 -verify=expected,nointerpreter %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,interpreter %s -fexperimental-new-constant-interpreter

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

void splash(Swim& swam) {                 // nointerpreter-note {{declared here}}
  static_assert(swam.phelps() == 28);     // ok
  static_assert((&swam)->phelps() == 28); // ok
  Swim* pswam = &swam;                    // expected-note {{declared here}}
  static_assert(pswam->phelps() == 28);   // expected-error {{static assertion expression is not an integral constant expression}} \
                                          // expected-note {{read of non-constexpr variable 'pswam' is not allowed in a constant expression}}
  static_assert(how_many(swam) == 28);    // ok
  static_assert(Swim().lochte() == 12);   // ok
  static_assert(swam.lochte() == 12);     // expected-error {{static assertion expression is not an integral constant expression}} \
                                          // nointerpreter-note {{virtual function called on object 'swam' whose dynamic type is not constant}}
  static_assert(swam.coughlin == 12);     // expected-error {{static assertion expression is not an integral constant expression}} \
                                          // nointerpreter-note {{read of variable 'swam' whose value is not known}}
}

extern Swim dc;
extern Swim& trident; // interpreter-note {{declared here}}

constexpr auto& sandeno   = typeid(dc);         // ok: can only be typeid(Swim)
constexpr auto& gallagher = typeid(trident);    // expected-error {{constexpr variable 'gallagher' must be initialized by a constant expression}} \
                                                // nointerpreter-note {{typeid applied to object 'trident' whose dynamic type is not constant}} \
                                                // interpreter-note {{initializer of 'trident' is unknown}}

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

namespace GH128409 {
  int &ff();
  int &x = ff(); // expected-note {{declared here}}
  constinit int &z = x; // expected-error {{variable does not have a constant initializer}} \
                        // expected-note {{required by 'constinit' specifier here}} \
                        // expected-note {{initializer of 'x' is not a constant expression}}
}

namespace GH129845 {
  int &ff();
  int &x = ff(); // expected-note {{declared here}}
  struct A { int& x; };
  constexpr A g = {x}; // expected-error {{constexpr variable 'g' must be initialized by a constant expression}} \
                       // expected-note {{initializer of 'x' is not a constant expression}}
  const A* gg = &g;
}

namespace extern_reference_used_as_unknown {
  extern int &x;
  int y;
  constinit int& g = (x,y); // expected-warning {{left operand of comma operator has no effect}}
}

namespace GH139452 {
struct Dummy {
  explicit operator bool() const noexcept { return true; }
};

struct Base { int error; };
struct Derived : virtual Base { };

template <class R>
constexpr R get_value() {
    const auto& derived_val = Derived{};
    if (derived_val.error != 0)
        /* nothing */;
    return R{};
}

int f() {
    return !get_value<Dummy>(); // contextually convert the function call result to bool
}
}

namespace uninit_reference_used {
  int y;
  constexpr int &r = r; // expected-error {{must be initialized by a constant expression}} \
  // expected-note {{initializer of 'r' is not a constant expression}} \
  // expected-note {{declared here}}
  constexpr int &rr = (rr, y);
  constexpr int &g() {
    int &x = x; // expected-warning {{reference 'x' is not yet bound to a value when used within its own initialization}} \
    // nointerpreter-note {{use of reference outside its lifetime is not allowed in a constant expression}} \
    // interpreter-note {{read of uninitialized object is not allowed in a constant expression}}
    return x;
  }
  constexpr int &gg = g(); // expected-error {{must be initialized by a constant expression}} \
  // expected-note {{in call to 'g()'}}
  constexpr int g2() {
    int &x = x; // expected-warning {{reference 'x' is not yet bound to a value when used within its own initialization}} \
    // nointerpreter-note {{use of reference outside its lifetime is not allowed in a constant expression}} \
    // interpreter-note {{read of uninitialized object is not allowed in a constant expression}}
    return x;
  }
  constexpr int gg2 = g2(); // expected-error {{must be initialized by a constant expression}} \
  // expected-note {{in call to 'g2()'}}
  constexpr int &g3() {
    int &x = (x,y); // expected-warning{{left operand of comma operator has no effect}} \
    // expected-warning {{reference 'x' is not yet bound to a value when used within its own initialization}} \
    // nointerpreter-note {{use of reference outside its lifetime is not allowed in a constant expression}}
    return x;
  }
  constexpr int &gg3 = g3(); // nointerpreter-error {{must be initialized by a constant expression}} \
  // nointerpreter-note {{in call to 'g3()'}}
  typedef decltype(sizeof(1)) uintptr_t;
  constexpr uintptr_t g4() {
    uintptr_t * &x = x; // expected-warning {{reference 'x' is not yet bound to a value when used within its own initialization}} \
    // nointerpreter-note {{use of reference outside its lifetime is not allowed in a constant expression}} \
    // interpreter-note {{read of uninitialized object is not allowed in a constant expression}}
    *(uintptr_t*)x = 10;
    return 3;
  }
  constexpr uintptr_t gg4 = g4(); // expected-error {{must be initialized by a constant expression}} \
  // expected-note {{in call to 'g4()'}}
  constexpr int g5() {
    int &x = x; // expected-warning {{reference 'x' is not yet bound to a value when used within its own initialization}} \
    // nointerpreter-note {{use of reference outside its lifetime is not allowed in a constant expression}} \
    // interpreter-note {{read of uninitialized object is not allowed in a constant expression}}
    return 3;
  }
  constexpr uintptr_t gg5 = g5(); // expected-error {{must be initialized by a constant expression}} \
  // expected-note {{in call to 'g5()'}}

}

namespace param_reference {
  constexpr int arbitrary = -12345;
  constexpr void f(const int &x = arbitrary) { // nointerpreter-note 3 {{declared here}} interpreter-note {{declared here}}
    constexpr const int &v1 = x; // expected-error {{must be initialized by a constant expression}} \
    // expected-note {{reference to 'x' is not a constant expression}}
    constexpr const int &v2 = (x, arbitrary); // expected-warning {{left operand of comma operator has no effect}}
    constexpr int v3 = x; // expected-error {{must be initialized by a constant expression}} \
                          // nointerpreter-note {{read of variable 'x' whose value is not known}}
    static_assert(x==arbitrary); // expected-error {{static assertion expression is not an integral constant expression}} \
                                 // nointerpreter-note {{read of variable 'x' whose value is not known}}
    static_assert(&x - &x == 0);
  }
}

namespace dropped_note {
  extern int &x; // expected-note {{declared here}}
  constexpr int f() { return x; } // nointerpreter-note {{read of non-constexpr variable 'x'}} \
                                  // interpreter-note {{initializer of 'x' is unknown}}
  constexpr int y = f(); // expected-error {{constexpr variable 'y' must be initialized by a constant expression}} expected-note {{in call to 'f()'}}
}

namespace dynamic {
  struct A {virtual ~A();};
  struct B : A {};
  void f(A& a) {
    constexpr B* b = dynamic_cast<B*>(&a); // expected-error {{must be initialized by a constant expression}} \
                                           // nointerpreter-note {{dynamic_cast applied to object 'a' whose dynamic type is not constant}}
    constexpr void* b2 = dynamic_cast<void*>(&a); // expected-error {{must be initialized by a constant expression}} \
                                                  // nointerpreter-note {{dynamic_cast applied to object 'a' whose dynamic type is not constant}}
  }
}

namespace unsized_array {
  void f(int (&a)[], int (&b)[], int (&c)[4]) {
    constexpr int t1 = a - a;
    constexpr int t2 = a - b; // expected-error {{constexpr variable 't2' must be initialized by a constant expression}} \
                              // nointerpreter-note {{arithmetic involving unrelated objects '&a[0]' and '&b[0]' has unspecified value}} \
                              // interpreter-note {{arithmetic involving unrelated objects 'a' and 'b' has unspecified value}}
    constexpr int t3 = a - &c[2];  // expected-error {{constexpr variable 't3' must be initialized by a constant expression}} \
                              // nointerpreter-note {{arithmetic involving unrelated objects '&a[0]' and '&c[2]' has unspecified value}} \
                              // interpreter-note {{arithmetic involving unrelated objects 'a' and '*((char*)&c + 8)' has unspecified value}}
  }
}

namespace casting {
  struct A {};
  struct B : A {};
  struct C : A {};
  extern A &a; // interpreter-note {{declared here}}
  extern B &b; // expected-note {{declared here}} interpreter-note 2 {{declared here}}
  constexpr B &t1 = (B&)a; // expected-error {{must be initialized by a constant expression}} \
                           // nointerpreter-note {{cannot cast object of dynamic type 'A' to type 'B'}} \
                           // interpreter-note {{initializer of 'a' is unknown}}
  constexpr B &t2 = (B&)(A&)b; // expected-error {{must be initialized by a constant expression}} \
                               // nointerpreter-note {{initializer of 'b' is not a constant expression}} \
                               // interpreter-note {{initializer of 'b' is unknown}}
  // FIXME: interpreter incorrectly rejects.
  constexpr bool t3 = &b + 1 == &(B&)(A&)b; // interpreter-error {{must be initialized by a constant expression}} \
                                            // interpreter-note {{initializer of 'b' is unknown}}
  constexpr C &t4 = (C&)(A&)b; // expected-error {{must be initialized by a constant expression}} \
                               // nointerpreter-note {{cannot cast object of dynamic type 'B' to type 'C'}} \
                               // interpreter-note {{initializer of 'b' is unknown}}
}

namespace pointer_comparisons {
  extern int &extern_n; // interpreter-note 4 {{declared here}}
  extern int &extern_n2;
  constexpr int f1(bool b, int& n) {
    if (b) {
      return &extern_n == &n;
    }
    return f1(true, n);
  }
  // FIXME: interpreter incorrectly rejects; both sides are the same constexpr-unknown value.
  static_assert(f1(false, extern_n)); // interpreter-error {{static assertion expression is not an integral constant expression}} \
                                      // interpreter-note {{initializer of 'extern_n' is unknown}}
  static_assert(&extern_n != &extern_n2); // expected-error {{static assertion expression is not an integral constant expression}} \
                                          // nointerpreter-note {{comparison between pointers to unrelated objects '&extern_n' and '&extern_n2' has unspecified value}} \
                                          // interpreter-note {{initializer of 'extern_n' is unknown}}
  void f2(const int &n) {
    constexpr int x = &x == &n; // nointerpreter-error {{must be initialized by a constant expression}} \
                                // nointerpreter-note {{comparison between pointers to unrelated objects '&x' and '&n' has unspecified value}}
    // Distinct variables are not equal, even if they're local variables.
    constexpr int y = &x == &y;
    static_assert(!y);
  }
  constexpr int f3() {
    int x;
    return &x == &extern_n; // nointerpreter-note {{comparison between pointers to unrelated objects '&x' and '&extern_n' has unspecified value}} \
                            // interpreter-note {{initializer of 'extern_n' is unknown}}
  }
  static_assert(!f3()); // expected-error {{static assertion expression is not an integral constant expression}} \
                        // expected-note {{in call to 'f3()'}}
  constexpr int f4() {
    int *p = new int;
    bool b = p == &extern_n; // nointerpreter-note {{comparison between pointers to unrelated objects '&{*new int#0}' and '&extern_n' has unspecified value}} \
                             // interpreter-note {{initializer of 'extern_n' is unknown}}
    delete p;
    return b;
  }
  static_assert(!f4()); // expected-error {{static assertion expression is not an integral constant expression}} \
                        // expected-note {{in call to 'f4()'}}
}
