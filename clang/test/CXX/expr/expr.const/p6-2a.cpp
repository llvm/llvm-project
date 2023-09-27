// RUN: %clang_cc1 -std=c++2a -verify %s

constexpr int non_class = 42;
constexpr int arr_non_class[5] = {1, 2, 3};

struct A {
  int member = 1;
  constexpr ~A() { member = member + 1; }
};
constexpr A class_ = {};
constexpr A arr_class[5] = {{}, {}};

struct Mutable {
  mutable int member = 1; // expected-note {{declared here}}
  constexpr ~Mutable() { member = member + 1; } // expected-note {{read of mutable member}}
};
constexpr Mutable mut_member; // expected-error {{must have constant destruction}} expected-note {{in call}}

struct MutableStore {
  mutable int member = 1; // expected-note {{declared here}}
  constexpr ~MutableStore() { member = 2; } // expected-note {{assignment to mutable member}}
};
constexpr MutableStore mut_store; // expected-error {{must have constant destruction}} expected-note {{in call}}

// Note: the constant destruction rules disallow this example even though hcm.n is a const object.
struct MutableConst {
  struct HasConstMember {
    const int n = 4;
  };
  mutable HasConstMember hcm; // expected-note {{here}}
  constexpr ~MutableConst() {
    int q = hcm.n; // expected-note {{read of mutable}}
  }
};
constexpr MutableConst mc; // expected-error {{must have constant destruction}} expected-note {{in call}}

struct Temporary {
  int &&temp;
  constexpr ~Temporary() {
    int n = temp; // expected-note {{outside the expression that created the temporary}}
  }
};
constexpr Temporary t = {3}; // expected-error {{must have constant destruction}} expected-note {{created here}} expected-note {{in call}}

namespace P1073R3 {
consteval int f() { return 42; } // expected-note {{declared here}}
consteval auto g() { return f; }
consteval int h(int (*p)() = g()) { return p(); }
constexpr int r = h();
constexpr auto e = g(); // expected-error {{constexpr variable 'e' must be initialized by a constant expression}} \
                           expected-note {{pointer to a consteval declaration is not a constant expression}}
static_assert(r == 42);
} // namespace P1073R3
