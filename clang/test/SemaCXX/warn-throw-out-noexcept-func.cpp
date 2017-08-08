// RUN: %clang_cc1 %s  -fdelayed-template-parsing -fcxx-exceptions -fsyntax-only -Wexceptions -verify -std=c++11
struct A_ShouldDiag {
  ~A_ShouldDiag(); // implicitly noexcept(true)
};
A_ShouldDiag::~A_ShouldDiag() { // expected-note {{destructor has a implicit non-throwing exception specification}}
  throw 1; // expected-warning {{has a non-throwing exception specification but can still throw}}
}
struct B_ShouldDiag {
  int i;
  ~B_ShouldDiag() noexcept(true) {} //no disg, no throw stmt
};
struct R_ShouldDiag : A_ShouldDiag {
  B_ShouldDiag b;
  ~R_ShouldDiag() { // expected-note  {{destructor has a implicit non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};

struct M_ShouldNotDiag {
  B_ShouldDiag b;
  ~M_ShouldNotDiag() noexcept(false);
};

M_ShouldNotDiag::~M_ShouldNotDiag() noexcept(false) {
  throw 1;
}

struct N_ShouldDiag {
  B_ShouldDiag b;
  ~N_ShouldDiag(); //implicitly noexcept(true)
};

N_ShouldDiag::~N_ShouldDiag() { // expected-note  {{destructor has a implicit non-throwing exception specification}}
  throw 1; // expected-warning {{has a non-throwing exception specification but}}
}
struct X_ShouldDiag {
  B_ShouldDiag b;
  ~X_ShouldDiag() noexcept { // expected-note  {{destructor has a non-throwing exception}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
struct Y_ShouldDiag : A_ShouldDiag {
  ~Y_ShouldDiag() noexcept(true) { // expected-note  {{destructor has a non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
struct C_ShouldNotDiag {
  int i;
  ~C_ShouldNotDiag() noexcept(false) {}
};
struct D_ShouldNotDiag {
  C_ShouldNotDiag c;
  ~D_ShouldNotDiag() { //implicitly noexcept(false)
    throw 1;
  }
};
struct E_ShouldNotDiag {
  C_ShouldNotDiag c;
  ~E_ShouldNotDiag(); //implicitly noexcept(false)
};
E_ShouldNotDiag::~E_ShouldNotDiag() //implicitly noexcept(false)
{
  throw 1;
}

template <typename T>
class A1_ShouldDiag {
  T b;

public:
  ~A1_ShouldDiag() { // expected-note  {{destructor has a implicit non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
template <typename T>
struct B1_ShouldDiag {
  T i;
  ~B1_ShouldDiag() noexcept(true) {}
};
template <typename T>
struct R1_ShouldDiag : A1_ShouldDiag<T> //expected-note {{in instantiation of member function}}
{
  B1_ShouldDiag<T> b;
  ~R1_ShouldDiag() { // expected-note  {{destructor has a implicit non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
template <typename T>
struct S1_ShouldDiag : A1_ShouldDiag<T> {
  B1_ShouldDiag<T> b;
  ~S1_ShouldDiag() noexcept { // expected-note  {{destructor has a non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
void operator delete(void *ptr) noexcept { // expected-note  {{deallocator has a non-throwing exception specification}}
  throw 1; // expected-warning {{has a non-throwing exception specification but}}
}
struct except_fun {
  static const bool i = false;
};
struct noexcept_fun {
  static const bool i = true;
};
template <typename T>
struct dependent_warn {
  ~dependent_warn() noexcept(T::i) {
    throw 1;
  }
};
template <typename T>
struct dependent_warn_noexcept {
  ~dependent_warn_noexcept() noexcept(T::i) { // expected-note  {{destructor has a non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
template <typename T>
struct dependent_warn_both {
  ~dependent_warn_both() noexcept(T::i) { // expected-note  {{destructor has a non-throwing exception specification}}
    throw 1; // expected-warning {{has a non-throwing exception specification but}}
  }
};
void foo() noexcept { //expected-note {{function declared non-throwing here}}
  throw 1; // expected-warning {{has a non-throwing exception specification but}}
}
struct Throws {
  ~Throws() noexcept(false);
};

struct ShouldDiagnose {
  Throws T;
  ~ShouldDiagnose() noexcept { //expected-note {{destructor has a non-throwing exception specification}}
    throw; // expected-warning {{has a non-throwing exception specification but}}
  }
};
struct ShouldNotDiagnose {
  Throws T;
  ~ShouldNotDiagnose() {
    throw;
  }
};

void bar_ShouldNotDiag() noexcept {
  try {
    throw 1;
  } catch (...) {
  }
}
void f_ShouldNotDiag() noexcept {
  try {
    throw 12;
  } catch (int) {
  }
}
void g_ShouldNotDiag() noexcept {
  try {
    throw 12;
  } catch (...) {
  }
}

void h_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw 12; // expected-warning {{has a non-throwing exception specification but}}
  } catch (const char *) {
  }
}

void i_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw 12;
  } catch (int) {
    throw; // expected-warning {{has a non-throwing exception specification but}}
  }
}
void j_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw 12;
  } catch (int) {
    throw "haha"; // expected-warning {{has a non-throwing exception specification but}}
  }
}

void k_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw 12;
  } catch (...) {
    throw; // expected-warning {{has a non-throwing exception specification but}}
  }
}

void loo_ShouldDiag(int i) noexcept { //expected-note {{function declared non-throwing here}}
  if (i)
    try {
      throw 12;
    } catch (int) {
      throw "haha"; //expected-warning {{has a non-throwing exception specification but}}
    }
  i = 10;
}

void loo1_ShouldNotDiag() noexcept {
  if (0)
    throw 12;
}

void loo2_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  if (1)
    throw 12; // expected-warning {{has a non-throwing exception specification but}}
}
struct S {};

void l_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw S{}; //expected-warning {{has a non-throwing exception specification but}}
  } catch (S *s) {
  }
}

void m_ShouldNotDiag() noexcept {
  try {
    const S &s = S{};
    throw s;
  } catch (S s) {
  }
}
void n_ShouldNotDiag() noexcept {
  try {
    S s = S{};
    throw s;
  } catch (const S &s) {
  }
}
void o_ShouldDiag() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw; //expected-warning {{has a non-throwing exception specification but}}
  } catch (...) {
  }
}

#define NOEXCEPT noexcept
void with_macro() NOEXCEPT { //expected-note {{function declared non-throwing here}}
  throw 1; // expected-warning {{has a non-throwing exception specification but}}
}

void with_try_block() try {
  throw 2;
} catch (...) {
}

void with_try_block1() noexcept try { //expected-note {{function declared non-throwing here}}
  throw 2; // expected-warning {{has a non-throwing exception specification but}}
} catch (char *) {
}

namespace derived {
struct B {};
struct D: B {};
void goodPlain() noexcept {
  try {
    throw D();
  } catch (B) {}
}
void goodReference() noexcept {
  try {
    throw D();
  } catch (B &) {}
}
void goodPointer() noexcept {
  D d;
  try {
    throw &d;
  } catch (B *) {}
}
void badPlain() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw B(); // expected-warning {{'badPlain' has a non-throwing exception specification but can still throw}}
  } catch (D) {}
}
void badReference() noexcept { //expected-note {{function declared non-throwing here}}
  try {
    throw B(); // expected-warning {{'badReference' has a non-throwing exception specification but can still throw}}
  } catch (D &) {}
}
void badPointer() noexcept { //expected-note {{function declared non-throwing here}}
  B b;
  try {
    throw &b; // expected-warning {{'badPointer' has a non-throwing exception specification but can still throw}}
  } catch (D *) {}
}
}

int main() {
  R1_ShouldDiag<int> o; //expected-note {{in instantiation of member function}}
  S1_ShouldDiag<int> b; //expected-note {{in instantiation of member function}}
  dependent_warn<except_fun> f;
  dependent_warn_noexcept<noexcept_fun> f1; //expected-note {{in instantiation of member function}}
  dependent_warn_both<except_fun> f2;
  dependent_warn_both<noexcept_fun> f3; //expected-note {{in instantiation of member function}}
  ShouldDiagnose obj;
  ShouldNotDiagnose obj1;
}
