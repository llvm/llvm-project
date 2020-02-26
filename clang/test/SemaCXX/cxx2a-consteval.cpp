// RUN: %clang_cc1 -std=c++2a -fsyntax-only -Wno-unused-value %s -verify

typedef __SIZE_TYPE__ size_t;

namespace basic_sema {

consteval int f1(int i) {
  return i;
}

consteval constexpr int f2(int i) { 
  //expected-error@-1 {{cannot combine}}
  return i;
}

constexpr auto l_eval = [](int i) consteval {
// expected-note@-1+ {{declared here}}

  return i;
};

constexpr consteval int f3(int i) {
  //expected-error@-1 {{cannot combine}}
  return i;
}

struct A {
  consteval int f1(int i) const {
// expected-note@-1 {{declared here}}
    return i;
  }
  consteval A(int i);
  consteval A() = default;
  consteval ~A() = default; // expected-error {{destructor cannot be declared consteval}}
};

consteval struct B {}; // expected-error {{struct cannot be marked consteval}}

consteval typedef B b; // expected-error {{typedef cannot be consteval}}

consteval int redecl() {return 0;} // expected-note {{previous declaration is here}}
constexpr int redecl() {return 0;} // expected-error {{constexpr declaration of 'redecl' follows consteval declaration}}

consteval int i = 0; // expected-error {{consteval can only be used in function declarations}}

consteval int; // expected-error {{consteval can only be used in function declarations}}

consteval int f1() {} // expected-error {{no return statement in consteval function}}

struct C {
  C() {}
  ~C() {}
};

struct D {
  C c;
  consteval D() = default; // expected-error {{cannot be consteval}}
  consteval ~D() = default; // expected-error {{destructor cannot be declared consteval}}
};

struct E : C {
  consteval ~E() {} // expected-error {{cannot be declared consteval}}
};
}

consteval int main() { // expected-error {{'main' is not allowed to be declared consteval}}
  return 0;
}

consteval int f_eval(int i) {
// expected-note@-1+ {{declared here}}
  return i;
}

namespace taking_address {

using func_type = int(int);

func_type* p1 = (&f_eval);
// expected-error@-1 {{take address}}
func_type* p7 = __builtin_addressof(f_eval);
// expected-error@-1 {{take address}}

auto p = f_eval;
// expected-error@-1 {{take address}}

auto m1 = &basic_sema::A::f1;
// expected-error@-1 {{take address}}
auto l1 = &decltype(basic_sema::l_eval)::operator();
// expected-error@-1 {{take address}}

consteval int f(int i) {
// expected-note@-1+ {{declared here}}
  return i;
}

auto ptr = &f;
// expected-error@-1 {{take address}}

auto f1() {
  return &f;
// expected-error@-1 {{take address}}
}

}

namespace invalid_function {

struct A {
  consteval void *operator new(size_t count);
  // expected-error@-1 {{'operator new' cannot be declared consteval}}
  consteval void *operator new[](size_t count);
  // expected-error@-1 {{'operator new[]' cannot be declared consteval}}
  consteval void operator delete(void* ptr);
  // expected-error@-1 {{'operator delete' cannot be declared consteval}}
  consteval void operator delete[](void* ptr);
  // expected-error@-1 {{'operator delete[]' cannot be declared consteval}}
  consteval ~A() {}
  // expected-error@-1 {{destructor cannot be declared consteval}}
};

}

namespace nested {
consteval int f() {
  return 0;
}

consteval int f1(...) {
  return 1;
}

enum E {};

using T = int(&)();

consteval auto operator+ (E, int(*a)()) {
  return 0;
}

void d() {
  auto i = f1(E() + &f);
}

auto l0 = [](auto) consteval {
  return 0;
};

int i0 = l0(&f1);

int i1 = f1(l0(4));

int i2 = f1(&f1, &f1, &f1, &f1, &f1, &f1, &f1);

int i3 = f1(f1(f1(&f1, &f1), f1(&f1, &f1), f1(f1(&f1, &f1), &f1)));

}

namespace user_defined_literal {

consteval int operator"" _test(unsigned long long i) {
// expected-note@-1+ {{declared here}}
  return 0;
}

int i = 0_test;

auto ptr = &operator"" _test;
// expected-error@-1 {{take address}}

consteval auto operator"" _test1(unsigned long long i) {
  return &f_eval;
}

auto i1 = 0_test1; // expected-error {{is not a constant expression}}
// expected-note@-1 {{is not a constant expression}}

}

namespace return_address {

consteval int f() {
// expected-note@-1 {{declared here}}
  return 0;
}

consteval int(*ret1(int i))() {
  return &f;
}

auto ptr = ret1(0);
// expected-error@-1 {{is not a constant expression}}
// expected-note@-2 {{pointer to a consteval}}

struct A {
  consteval int f(int) {
    // expected-note@-1+ {{declared here}}
    return 0;    
  }
};

using mem_ptr_type = int (A::*)(int);

template<mem_ptr_type ptr>
struct C {};

C<&A::f> c;
// expected-error@-1 {{is not a constant expression}}
// expected-note@-2 {{pointer to a consteval}}

consteval mem_ptr_type ret2() {
  return &A::f;
}

C<ret2()> c1;
// expected-error@-1 {{is not a constant expression}}
// expected-note@-2 {{pointer to a consteval}}

}

namespace context {

int g_i;
// expected-note@-1 {{declared here}}

consteval int f(int) {
  return 0;
}

constexpr int c_i = 0;

int t1 = f(g_i);
// expected-error@-1 {{is not a constant expression}}
// expected-note@-2 {{read of non-const variable}}
int t3 = f(c_i);

constexpr int f_c(int i) {
// expected-note@-1 {{declared here}}
  int t = f(i);
// expected-error@-1 {{is not a constant expression}}
// expected-note@-2 {{read of non-const variable}}
  return f(0);  
}

consteval int f_eval(int i) {
  return f(i);
}

auto l0 = [](int i) consteval {
  return f(i);
};

auto l1 = [](int i) constexpr {
// expected-note@-1 {{declared here}}
  int t = f(i);
// expected-error@-1 {{is not a constant expression}}
// expected-note@-2 {{read of non-const variable}}
  return f(0);  
};

}

namespace temporaries {

struct A {
  consteval int ret_i() const { return 0; }
  consteval A ret_a() const { return A{}; }
  constexpr ~A() { }
};

consteval int by_value_a(A a) { return a.ret_i(); }

consteval int const_a_ref(const A &a) {
  return a.ret_i();
}

consteval int rvalue_ref(const A &&a) {
  return a.ret_i();
}

consteval const A &to_lvalue_ref(const A &&a) {
  return a;
}

void test() {
  constexpr A a {};
  { int k = A().ret_i(); }
  { A k = A().ret_a(); }
  { A k = to_lvalue_ref(A()); }// expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}} expected-note@-1 {{temporary created here}}
  { A k = to_lvalue_ref(A().ret_a()); } // expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}} expected-note@-1 {{temporary created here}}
  { int k = A().ret_a().ret_i(); }
  { int k = by_value_a(A()); }
  { int k = const_a_ref(A()); }
  { int k = const_a_ref(a); }
  { int k = rvalue_ref(A()); }
  { int k = rvalue_ref(static_cast<const A&&>(a)); }
  { int k = const_a_ref(A().ret_a()); }
  { int k = const_a_ref(to_lvalue_ref(A().ret_a())); }
  { int k = const_a_ref(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = by_value_a(A().ret_a()); }
  { int k = by_value_a(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = (A().ret_a(), A().ret_i()); }
  { int k = (const_a_ref(A().ret_a()), A().ret_i()); }//
}

}

namespace alloc {

consteval int f() {
  int *A = new int(0);
// expected-note@-1+ {{allocation performed here was not deallocated}}
  return *A;
}

int i1 = f(); // expected-error {{is not a constant expression}}

struct A {
  int* p = new int(42);
  // expected-note@-1+ {{heap allocation performed here}}
  consteval int ret_i() const { return p ? *p : 0; }
  consteval A ret_a() const { return A{}; }
  constexpr ~A() { delete p; }
};

consteval int by_value_a(A a) { return a.ret_i(); }

consteval int const_a_ref(const A &a) {
  return a.ret_i();
}

consteval int rvalue_ref(const A &&a) {
  return a.ret_i();
}

consteval const A &to_lvalue_ref(const A &&a) {
  return a;
}

void test() {
  constexpr A a{ nullptr };
  { int k = A().ret_i(); }
  { A k = A().ret_a(); } // expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}}
  { A k = to_lvalue_ref(A()); } // expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}} expected-note@-1 {{temporary created here}}
  { A k = to_lvalue_ref(A().ret_a()); } // expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}} expected-note@-1 {{temporary created here}}
  { int k = A().ret_a().ret_i(); }
  { int k = by_value_a(A()); }
  { int k = const_a_ref(A()); }
  { int k = const_a_ref(a); }
  { int k = rvalue_ref(A()); }
  { int k = rvalue_ref(static_cast<const A&&>(a)); }
  { int k = const_a_ref(A().ret_a()); }
  { int k = const_a_ref(to_lvalue_ref(A().ret_a())); }
  { int k = const_a_ref(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = by_value_a(A().ret_a()); }
  { int k = by_value_a(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = (A().ret_a(), A().ret_i()); }// expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}}
  { int k = (const_a_ref(A().ret_a()), A().ret_i()); }
}

}

namespace self_referencing {

struct S {
  S* ptr = nullptr;
  constexpr S(int i) : ptr(this) {
    if (this == ptr && i)
      ptr = nullptr;
  }
  constexpr ~S() {}
};

consteval S f(int i) {
  return S(i);
}

void test() {
  S s(1);
  s = f(1);
  s = f(0); // expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}} expected-note@-1 {{temporary created here}}
}

}
