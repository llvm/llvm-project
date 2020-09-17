// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

// A constexpr specifier used in an object declaration declares the object as
// const.
constexpr int a = 0;
extern const int a;

int i; // expected-note 2{{here}}
constexpr int *b = &i;
extern int *const b;

constexpr int &c = i;
extern int &c;

constexpr int (*d)(int) = 0;
extern int (*const d)(int);

// A variable declaration which uses the constexpr specifier shall have an
// initializer and shall be initialized by a constant expression.
constexpr int ni1; // expected-error {{default initialization of an object of const type 'const int'}}
constexpr struct C { C(); } ni2; // expected-error {{cannot have non-literal type 'const struct C'}} expected-note 3{{has no constexpr constructors}}
constexpr double &ni3; // expected-error {{declaration of reference variable 'ni3' requires an initializer}}

constexpr int nc1 = i; // expected-error {{constexpr variable 'nc1' must be initialized by a constant expression}} expected-note {{read of non-const variable 'i' is not allowed in a constant expression}}
constexpr C nc2 = C(); // expected-error {{cannot have non-literal type 'const C'}}
int &f(); // expected-note 2{{declared here}}
constexpr int &nc3 = f(); // expected-error {{constexpr variable 'nc3' must be initialized by a constant expression}} expected-note {{non-constexpr function 'f' cannot be used in a constant expression}}
constexpr int nc4(i); // expected-error {{constexpr variable 'nc4' must be initialized by a constant expression}} expected-note {{read of non-const variable 'i' is not allowed in a constant expression}}
constexpr C nc5((C())); // expected-error {{cannot have non-literal type 'const C'}}
constexpr int &nc6(f()); // expected-error {{constexpr variable 'nc6' must be initialized by a constant expression}} expected-note {{non-constexpr function 'f'}}

struct pixel {
  int x, y;
};
constexpr pixel ur = { 1294, 1024 }; // ok
constexpr pixel origin;              // expected-error {{default initialization of an object of const type 'const pixel' without a user-provided default constructor}}

#if __cplusplus > 201702L
// A constexpr variable shall have constant destruction.
struct A {
  bool ok;
  constexpr A(bool ok) : ok(ok) {}
  constexpr ~A() noexcept(false) {
    void oops(); // expected-note 2{{declared here}}
    if (!ok) oops(); // expected-note 2{{non-constexpr function}}
  }
};

constexpr A const_dtor(true);
constexpr A non_const_dtor(false); // expected-error {{must have constant destruction}} expected-note {{in call}}
constexpr A arr_dtor[5] = {true, true, true, false, true}; // expected-error {{must have constant destruction}} expected-note {{in call to '&arr_dtor[3]->~A()'}}
#endif
