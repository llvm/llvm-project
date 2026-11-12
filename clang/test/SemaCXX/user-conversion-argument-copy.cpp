// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx11- %s
// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx11- -fms-compatibility %s

// When a constructor is called as a user-defined conversion, copying its
// argument into its parameter first looks for a constructor that needs no
// further user-defined conversion. That must select the same constructor as
// overload resolution over all of them.

namespace nested_converting_constructor {
struct X;
struct Y {
  constexpr Y(const X &) {}
};
struct X {
  int Ctor;
  constexpr X() : Ctor(0) {}
  X(X &);
  constexpr X(Y) : Ctor(1) {}
};
struct Derived : X {
  constexpr Derived() {}
};
struct C {
  int Ctor;
  constexpr C(X x) : Ctor(x.Ctor) {}
};
constexpr int by_argument(C c) { return c.Ctor; }
constexpr C by_return(const X &x) { return x; }

constexpr X x;
constexpr Derived derived;
constexpr C by_copy_init = x;
constexpr const C &by_reference = x;

static_assert(by_copy_init.Ctor == 1, "");
static_assert(by_reference.Ctor == 1, "");
static_assert(by_argument(x) == 1, "");
static_assert(by_argument(static_cast<const X &&>(x)) == 1, "");
static_assert(by_argument(derived) == 1, "");
static_assert(by_return(x).Ctor == 1, "");
} // namespace nested_converting_constructor

namespace nested_conversion_function {
struct Ref {};
struct Ptr {
  int Ctor;
  constexpr Ptr() : Ctor(0) {}
  Ptr(Ptr &);
  constexpr Ptr(Ref) : Ctor(1) {}
  constexpr operator Ref() const { return Ref(); }
};
struct Holder {
  int Ctor;
  constexpr Holder(Ptr p) : Ctor(p.Ctor) {}
};
constexpr int by_argument(Holder h) { return h.Ctor; }

constexpr Ptr ptr;
constexpr Holder by_copy_init = ptr;

static_assert(by_copy_init.Ctor == 1, "");
static_assert(by_argument(ptr) == 1, "");
static_assert(by_argument(static_cast<const Ptr &&>(ptr)) == 1, "");

Holder from_prvalue = Ptr();
} // namespace nested_conversion_function

namespace user_conversion_is_better_than_ellipsis {
struct X;
struct Y {
  constexpr Y(const X &) {}
};
struct X {
  int Ctor;
  constexpr X(int) : Ctor(0) {}
  X(X &);
  constexpr X(Y) : Ctor(1) {}
  constexpr X(...) : Ctor(2) {}
};
struct C {
  int Ctor;
  constexpr C(X x) : Ctor(x.Ctor) {}
};

constexpr X x(0);
constexpr C c = x;

static_assert(c.Ctor == 1, "");
} // namespace user_conversion_is_better_than_ellipsis

namespace conversion_function_to_base_is_not_used {
struct Base {
  int Ctor;
  constexpr Base() : Ctor(0) {}
  constexpr Base(const Base &) : Ctor(1) {}
};
struct Derived : Base {
  constexpr Derived() {}
  constexpr operator Base() const { return Base(); } // expected-warning {{will never be used}}
};
struct C {
  int Ctor;
  constexpr C(Base b) : Ctor(b.Ctor) {}
};

constexpr Derived derived;
constexpr C c = derived;

static_assert(c.Ctor == 1, "");
} // namespace conversion_function_to_base_is_not_used

namespace deleted_copy_constructor {
struct X;
struct Y {
  Y(const X &);
};
struct X {
  X();
  X(const X &) = delete; // expected-note {{'X' has been explicitly marked deleted here}}
  X(Y);
};
struct C {
  C(X); // expected-note {{passing argument to parameter here}}
};
void f(C);

void test(X x) {
  f(x); // expected-error {{call to deleted constructor of 'X'}}
}
} // namespace deleted_copy_constructor

namespace ambiguous_copy_constructors {
struct X;
struct Y {
  Y(const X &);
};
struct X {
  X();
  X(const X &, int = 0);  // expected-note {{candidate constructor}}
  X(const X &, long = 0); // expected-note {{candidate constructor}}
  X(Y);
};
struct C {
  C(X); // expected-note {{passing argument to parameter here}}
};
void f(C);

void test(X x) {
  f(x); // expected-error {{call to constructor of 'X' is ambiguous}}
}
} // namespace ambiguous_copy_constructors

namespace no_viable_constructor {
struct X {
  X();   // expected-note {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  X(X &); // expected-note {{candidate constructor not viable: 1st argument ('const X') would lose const qualifier}}
};
struct C {
  C(X); // expected-note {{passing argument to parameter here}}
};
void f(C);

void test(const X &x) {
  f(x); // expected-error {{no matching constructor for initialization of 'X'}}
}
} // namespace no_viable_constructor
