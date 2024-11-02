// RUN: %clang_cc1 -std=c++2a -emit-llvm-only -Wno-unused-value -Wno-vla %s -verify

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
  consteval D() = default; // expected-error {{cannot be marked consteval}}
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
// expected-note@-2 {{function parameter}}
  return f(0);
}

consteval int f_eval(int i) {
  return f(i);
}

auto l0 = [](int i) consteval {
  return f(i);
};

auto l1 = [](int i) constexpr { // expected-error{{cannot take address of immediate call operator}} \
                                // expected-note {{declared here}}
  int t = f(i);
  return f(0);
};

int(*test)(int)  = l1;

}

namespace consteval_lambda_in_template {
struct S {
    int *value;
    constexpr S(int v) : value(new int {v}) {}
    constexpr ~S() { delete value; }
};
consteval S fn() { return S(5); }

template <typename T>
void fn2() {
    (void)[]() consteval -> int {
      return *(fn().value);  // OK, immediate context
    };
}

void caller() {
    fn2<int>();
}
}

namespace std {

template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T &> { using type = T; };
template <typename T> struct remove_reference<T &&> { using type = T; };

template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T &&t) noexcept {
  return static_cast<typename std::remove_reference<T>::type &&>(t);
}

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
  { int k = rvalue_ref(std::move(a)); }
  { int k = const_a_ref(A().ret_a()); }
  { int k = const_a_ref(to_lvalue_ref(A().ret_a())); }
  { int k = const_a_ref(to_lvalue_ref(std::move(a))); }
  { int k = by_value_a(A().ret_a()); }
  { int k = by_value_a(to_lvalue_ref(std::move(a))); }
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
  { A k = to_lvalue_ref(A().ret_a()); }
  // expected-error@-1 {{'alloc::A::ret_a' is not a constant expression}}
  // expected-note@-2 {{heap-allocated object is not a constant expression}}
  // expected-error@-3 {{'alloc::to_lvalue_ref' is not a constant expression}}
  // expected-note@-4 {{reference to temporary is not a constant expression}}
  // expected-note@-5 {{temporary created here}}
  { int k = A().ret_a().ret_i(); }
  // expected-error@-1 {{'alloc::A::ret_a' is not a constant expression}}
  // expected-note@-2 {{heap-allocated object is not a constant expression}}
  { int k = by_value_a(A()); }
  { int k = const_a_ref(A()); }
  { int k = const_a_ref(a); }
  { int k = rvalue_ref(A()); }
  { int k = rvalue_ref(std::move(a)); }
  { int k = const_a_ref(A().ret_a()); }
  // expected-error@-1 {{'alloc::A::ret_a' is not a constant expression}}
  // expected-note@-2 {{is not a constant expression}}
  { int k = const_a_ref(to_lvalue_ref(A().ret_a())); }
  // expected-error@-1 {{'alloc::A::ret_a' is not a constant expression}}
  // expected-note@-2 {{is not a constant expression}}
  { int k = const_a_ref(to_lvalue_ref(std::move(a))); }
  { int k = by_value_a(A().ret_a()); }
  { int k = by_value_a(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = (A().ret_a(), A().ret_i()); }// expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}}
  { int k = (const_a_ref(A().ret_a()), A().ret_i()); }
  // expected-error@-1 {{'alloc::A::ret_a' is not a constant expression}}
  // expected-note@-2 {{is not a constant expression}}
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

struct S1 {
  S1* ptr = nullptr;
  consteval S1(int i) : ptr(this) {
    if (this == ptr && i)
      ptr = nullptr;
  }
  constexpr ~S1() {}
};

void test1() {
  S1 s(1);
  s = S1(1);
  s = S1(0); // expected-error {{is not a constant expression}}
  // expected-note@-1 {{is not a constant expression}} expected-note@-1 {{temporary created here}}
}

}
namespace ctor {

consteval int f_eval() { // expected-note+ {{declared here}}
  return 0;
}

namespace std {
  struct strong_ordering {
    int n;
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
  constexpr bool operator!=(strong_ordering, int);
}

namespace override {
  struct A {
    virtual consteval void f(); // expected-note {{overridden}}
    virtual void g(); // expected-note {{overridden}}
  };
  struct B : A {
    consteval void f();
    void g();
  };
  struct C : A {
    void f(); // expected-error {{non-consteval function 'f' cannot override a consteval function}}
    consteval void g(); // expected-error {{consteval function 'g' cannot override a non-consteval function}}
  };

  namespace implicit_equals_1 {
    struct Y;
    struct X {
      std::strong_ordering operator<=>(const X&) const;
      constexpr bool operator==(const X&) const;
      virtual consteval bool operator==(const Y&) const; // expected-note {{here}}
    };
    struct Y : X {
      std::strong_ordering operator<=>(const Y&) const = default;
      // expected-error@-1 {{non-consteval function 'operator==' cannot override a consteval function}}
    };
  }

  namespace implicit_equals_2 {
    struct Y;
    struct X {
      constexpr std::strong_ordering operator<=>(const X&) const;
      constexpr bool operator==(const X&) const;
      virtual bool operator==(const Y&) const; // expected-note {{here}}
    };
    struct Y : X {
      consteval std::strong_ordering operator<=>(const Y&) const = default;
      // expected-error@-1 {{consteval function 'operator==' cannot override a non-consteval function}}
    };
  }
}

namespace operator_rewrite {
  struct A {
    friend consteval int operator<=>(const A&, const A&) { return 0; }
  };
  const bool k = A() < A();
  static_assert(!k);

  A a;
  bool k2 = A() < a; // OK, does not access 'a'.

  struct B {
    friend consteval int operator<=>(const B &l, const B &r) { return r.n - l.n; } // expected-note {{read of }}
    int n;
  };
  static_assert(B() >= B());
  B b; // expected-note {{here}}
  bool k3 = B() < b; // expected-error-re {{call to consteval function '{{.*}}::operator<=>' is not a constant expression}} expected-note {{in call}}
}

struct A {
  int(*ptr)();
  consteval A(int(*p)() = nullptr) : ptr(p) {}
};

struct B {
  int(*ptr)();
  B() : ptr(nullptr) {}
  consteval B(int(*p)(), int) : ptr(p) {}
};

void test() {
  { A a; }
  { A a(&f_eval); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B b(nullptr, 0); }
  { B b(&f_eval, 0); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A a{}; }
  { A a{&f_eval}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B b{nullptr, 0}; }
  { B b{&f_eval, 0}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A a = A(); }
  { A a = A(&f_eval); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B b = B(nullptr, 0); }
  { B b = B(&f_eval, 0); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A a = A{}; }
  { A a = A{&f_eval}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B b = B{nullptr, 0}; }
  { B b = B{&f_eval, 0}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A a; a = A(); }
  { A a; a = A(&f_eval); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B b; b = B(nullptr, 0); }
  { B b; b = B(&f_eval, 0); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A a; a = A{}; }
  { A a; a = A{&f_eval}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B b; b = B{nullptr, 0}; }
  { B b; b = B{&f_eval, 0}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A* a; a = new A(); }
  { A* a; a = new A(&f_eval); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B* b; b = new B(nullptr, 0); }
  { B* b; b = new B(&f_eval, 0); } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { A* a; a = new A{}; }
  { A* a; a = new A{&f_eval}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { B* b; b = new B{nullptr, 0}; }
  { B* b; b = new B{&f_eval, 0}; } // expected-error {{is not a constant expression}} expected-note {{to a consteval}}
}

}

namespace copy_ctor {

consteval int f_eval() { // expected-note+ {{declared here}}
  return 0;
}

struct Copy {
  int(*ptr)();
  constexpr Copy(int(*p)() = nullptr) : ptr(p) {}
  consteval Copy(const Copy&) = default;
};

constexpr const Copy &to_lvalue_ref(const Copy &&a) {
  return a;
}

void test() {
  constexpr const Copy C;
  // there is no the copy constructor call when its argument is a prvalue because of garanteed copy elision.
  // so we need to test with both prvalue and xvalues.
  { Copy c(C); }
  { Copy c((Copy(&f_eval))); }// expected-error {{cannot take address of consteval}}
  { Copy c(std::move(C)); }
  { Copy c(std::move(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c(to_lvalue_ref((Copy(&f_eval)))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c(to_lvalue_ref(std::move(C))); }
  { Copy c(to_lvalue_ref(std::move(Copy(&f_eval)))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c = Copy(C); }
  { Copy c = Copy(Copy(&f_eval)); }// expected-error {{cannot take address of consteval}}
  { Copy c = Copy(std::move(C)); }
  { Copy c = Copy(std::move(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c = Copy(to_lvalue_ref(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c = Copy(to_lvalue_ref(std::move(C))); }
  { Copy c = Copy(to_lvalue_ref(std::move(Copy(&f_eval)))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c; c = Copy(C); }
  { Copy c; c = Copy(Copy(&f_eval)); }// expected-error {{cannot take address of consteval}}
  { Copy c; c = Copy(std::move(C)); }
  { Copy c; c = Copy(std::move(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c; c = Copy(to_lvalue_ref(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy c; c = Copy(to_lvalue_ref(std::move(C))); }
  { Copy c; c = Copy(to_lvalue_ref(std::move(Copy(&f_eval)))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy* c; c = new Copy(C); }
  { Copy* c; c = new Copy(Copy(&f_eval)); }// expected-error {{cannot take address of consteval}}
  { Copy* c; c = new Copy(std::move(C)); }
  { Copy* c; c = new Copy(std::move(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy* c; c = new Copy(to_lvalue_ref(Copy(&f_eval))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
  { Copy* c; c = new Copy(to_lvalue_ref(std::move(C))); }
  { Copy* c; c = new Copy(to_lvalue_ref(std::move(Copy(&f_eval)))); }// expected-error {{is not a constant expression}} expected-note {{to a consteval}}
}

} // namespace special_ctor

namespace unevaluated {

template <typename T, typename U> struct is_same { static const bool value = false; };
template <typename T> struct is_same<T, T> { static const bool value = true; };

long f(); // expected-note {{declared here}}
auto consteval g(auto a) {
  return a;
}

auto e = g(f()); // expected-error {{is not a constant expression}}
                 // expected-note@-1 {{non-constexpr function 'f' cannot be used in a constant expression}}

using T = decltype(g(f()));
static_assert(is_same<long, T>::value);

} // namespace unevaluated

namespace value_dependent {

consteval int foo(int x) {
  return x;
}

template <int X> constexpr int bar() {
  // Previously this call was rejected as value-dependent constant expressions
  // can't be immediately evaluated. Now we show that we don't immediately
  // evaluate them until they are instantiated.
  return foo(X);
}

template <typename T> constexpr int baz() {
  constexpr int t = sizeof(T);
  // Previously this call was rejected as `t` is value-dependent and its value
  // is unknown until the function is instantiated. Now we show that we don't
  // reject such calls.
  return foo(t);
}

static_assert(bar<15>() == 15);
static_assert(baz<int>() == sizeof(int));
} // namespace value_dependent

// https://github.com/llvm/llvm-project/issues/55601
namespace issue_55601 {
template<typename T>
class Bar {
  consteval static T x() { return 5; }  // expected-note {{non-constexpr constructor 'derp' cannot be used in a constant expression}}
 public:
  Bar() : a(x()) {} // expected-error {{call to consteval function 'issue_55601::Bar<issue_55601::derp>::x' is not a constant expression}}
                    // expected-error@-1 {{call to consteval function 'issue_55601::derp::operator int' is not a constant expression}}
                    // expected-note@-2 {{in call to 'x()'}}
                    // expected-note@-3 {{non-literal type 'issue_55601::derp' cannot be used in a constant expression}}
 private:
  int a;
};
Bar<int> f;
Bar<float> g;

struct derp {
  // Can't be used in a constant expression
  derp(int); // expected-note {{declared here}}
  consteval operator int() const { return 5; }
};
Bar<derp> a; // expected-note {{in instantiation of member function 'issue_55601::Bar<issue_55601::derp>::Bar' requested here}}

struct constantDerp {
  // Can be used in a constant expression.
  consteval constantDerp(int) {}
  consteval operator int() const { return 5; }
};
Bar<constantDerp> b;

} // namespace issue_55601

namespace default_argument {

// Previously calls of consteval functions in default arguments were rejected.
// Now we show that we don't reject such calls.
consteval int foo() { return 1; }
consteval int bar(int i = foo()) { return i * i; }

struct Test1 {
  Test1(int i = bar(13)) {}
  void v(int i = bar(13) * 2 + bar(15)) {}
};
Test1 t1;

struct Test2 {
  constexpr Test2(int i = bar()) {}
  constexpr void v(int i = bar(bar(bar(foo())))) {}
};
Test2 t2;

} // namespace default_argument

namespace PR50779 {
struct derp {
  int b = 0;
};

constexpr derp d;

struct test {
  consteval int operator[](int i) const { return {}; }
  consteval const derp * operator->() const { return &d; }
  consteval int f() const { return 12; } // expected-note 2{{declared here}}
};

constexpr test a;

// We previously rejected both of these overloaded operators as taking the
// address of a consteval function outside of an immediate context, but we
// accepted direct calls to the overloaded operator. Now we show that we accept
// both forms.
constexpr int s = a.operator[](1);
constexpr int t = a[1];
constexpr int u = a.operator->()->b;
constexpr int v = a->b;
// FIXME: I believe this case should work, but we currently reject.
constexpr int w = (a.*&test::f)(); // expected-error {{cannot take address of consteval function 'f' outside of an immediate invocation}}
constexpr int x = a.f();

// Show that we reject when not in an immediate context.
int w2 = (a.*&test::f)(); // expected-error {{cannot take address of consteval function 'f' outside of an immediate invocation}}
}

namespace PR48235 {
consteval int d() {
  return 1;
}

struct A {
  consteval int a() const { return 1; }

  void b() {
    this->a() + d(); // expected-error {{call to consteval function 'PR48235::A::a' is not a constant expression}} \
                     // expected-note {{use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}}
  }

  void c() {
    a() + d(); // expected-error {{call to consteval function 'PR48235::A::a' is not a constant expression}} \
               // expected-note {{use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}}
  }
};
} // PR48235

namespace NamespaceScopeConsteval {
struct S {
  int Val; // expected-note {{subobject declared here}}
  consteval S() {}
};

S s1; // expected-error {{call to consteval function 'NamespaceScopeConsteval::S::S' is not a constant expression}} \
         expected-note {{subobject 'Val' is not initialized}}

template <typename Ty>
struct T {
  Ty Val; // expected-note {{subobject declared here}}
  consteval T() {}
};

T<int> t; // expected-error {{call to consteval function 'NamespaceScopeConsteval::T<int>::T' is not a constant expression}} \
             expected-note {{subobject 'Val' is not initialized}}

} // namespace NamespaceScopeConsteval

namespace Issue54578 {
// We expect the user-defined literal to be resovled entirely at compile time
// despite being instantiated through a template.
inline consteval unsigned char operator""_UC(const unsigned long long n) {
  return static_cast<unsigned char>(n);
}

inline constexpr char f1(const auto octet) {
  return 4_UC;
}

template <typename Ty>
inline constexpr char f2(const Ty octet) {
  return 4_UC;
}

void test() {
  static_assert(f1('a') == 4);
  static_assert(f2('a') == 4);
  constexpr int c = f1('a') + f2('a');
  static_assert(c == 8);
}
}

namespace defaulted_special_member_template {
template <typename T>
struct default_ctor {
  T data;
  consteval default_ctor() = default; // expected-note {{non-constexpr constructor 'foo' cannot be used in a constant expression}}
};

template <typename T>
struct copy {
  T data;

  consteval copy(const copy &) = default;            // expected-note {{non-constexpr constructor 'foo' cannot be used in a constant expression}}
  consteval copy &operator=(const copy &) = default; // expected-note {{non-constexpr function 'operator=' cannot be used in a constant expression}}
  copy() = default;
};

template <typename T>
struct move {
  T data;

  consteval move(move &&) = default;            // expected-note {{non-constexpr constructor 'foo' cannot be used in a constant expression}}
  consteval move &operator=(move &&) = default; // expected-note {{non-constexpr function 'operator=' cannot be used in a constant expression}}
  move() = default;
};

struct foo {
  foo() {}            // expected-note {{declared here}}
  foo(const foo &) {} // expected-note {{declared here}}
  foo(foo &&) {}      // expected-note {{declared here}}

  foo& operator=(const foo &) { return *this; } // expected-note {{declared here}}
  foo& operator=(foo &&) { return *this; }      // expected-note {{declared here}}
};

void func() {
  default_ctor<foo> fail0; // expected-error {{call to consteval function 'defaulted_special_member_template::default_ctor<defaulted_special_member_template::foo>::default_ctor' is not a constant expression}} \
                              expected-note {{in call to 'default_ctor()'}}

  copy<foo> good0;
  copy<foo> fail1{good0}; // expected-error {{call to consteval function 'defaulted_special_member_template::copy<defaulted_special_member_template::foo>::copy' is not a constant expression}} \
                             expected-note {{in call to 'copy(good0)'}}
  fail1 = good0;          // expected-error {{call to consteval function 'defaulted_special_member_template::copy<defaulted_special_member_template::foo>::operator=' is not a constant expression}} \
                             expected-note {{in call to 'fail1.operator=(good0)'}}

  move<foo> good1;
  move<foo> fail2{static_cast<move<foo>&&>(good1)}; // expected-error {{call to consteval function 'defaulted_special_member_template::move<defaulted_special_member_template::foo>::move' is not a constant expression}} \
                                                       expected-note {{in call to 'move(good1)'}}
  fail2 = static_cast<move<foo>&&>(good1);          // expected-error {{call to consteval function 'defaulted_special_member_template::move<defaulted_special_member_template::foo>::operator=' is not a constant expression}} \
                                                       expected-note {{in call to 'fail2.operator=(good1)'}}
}
} // namespace defaulted_special_member_template

namespace multiple_default_constructors {
struct Foo {
  Foo() {} // expected-note {{declared here}}
};
struct Bar {
  Bar() = default;
};
struct Baz {
  consteval Baz() {}
};

template <typename T, unsigned N>
struct S {
  T data;
  S() requires (N==1) = default;
  // This cannot be used in constexpr context.
  S() requires (N==2) {}  // expected-note {{declared here}}
  consteval S() requires (N==3) = default;  // expected-note {{non-constexpr constructor 'Foo' cannot be used in a constant expression}}
};

void func() {
  // Explictly defaulted constructor.
  S<Foo, 1> s1;
  S<Bar, 1> s2;
  // User provided constructor.
  S<Foo, 2> s3;
  S<Bar, 2> s4;
  // Consteval explictly defaulted constructor.
  S<Foo, 3> s5; // expected-error {{call to consteval function 'multiple_default_constructors::S<multiple_default_constructors::Foo, 3>::S' is not a constant expression}} \
                   expected-note {{in call to 'S()'}}
  S<Bar, 3> s6;
  S<Baz, 3> s7;
}

consteval int aConstevalFunction() { // expected-error {{consteval function never produces a constant expression}}
  // Defaulted default constructors are implicitly consteval.
  S<Bar, 1> s1;

  S<Baz, 2> s4; // expected-note {{non-constexpr constructor 'S' cannot be used in a constant expression}}

  S<Bar, 3> s2;
  S<Baz, 3> s3;
  return 0;
}

} // namespace multiple_default_constructors

namespace GH50055 {
enum E {e1=0, e2=1};
consteval int testDefaultArgForParam(E eParam = (E)-1) {
// expected-error@-1 {{integer value -1 is outside the valid range of values [0, 1] for the enumeration type 'E'}}
  return (int)eParam;
}

int test() {
  return testDefaultArgForParam() + testDefaultArgForParam((E)1);
}
}

namespace GH51182 {
// Nested consteval function.
consteval int f(int v) {
  return v;
}

template <typename T>
consteval int g(T a) {
  // An immediate function context.
  int n = f(a);
  return n;
}
static_assert(g(100) == 100);
// --------------------------------------
template <typename T>
consteval T max(const T& a, const T& b) {
    return (a > b) ? a : b;
}
template <typename T>
consteval T mid(const T& a, const T& b, const T& c) {
    T m = max(max(a, b), c);
    if (m == a)
        return max(b, c);
    if (m == b)
        return max(a, c);
    return max(a, b);
}
static_assert(max(1,2)==2);
static_assert(mid(1,2,3)==2);
} // namespace GH51182

// https://github.com/llvm/llvm-project/issues/56183
namespace GH56183 {
consteval auto Foo(auto c) { return c; }
consteval auto Bar(auto f) { return f(); }
void test() {
  constexpr auto x = Foo(Bar([] { return 'a'; }));
  static_assert(x == 'a');
}
}  // namespace GH56183

// https://github.com/llvm/llvm-project/issues/51695
namespace GH51695 {
// Original ========================================
template <typename T>
struct type_t {};

template <typename...>
struct list_t {};

template <typename T, typename... Ts>
consteval auto pop_front(list_t<T, Ts...>) -> auto {
  return list_t<Ts...>{};
}

template <typename... Ts, typename F>
consteval auto apply(list_t<Ts...>, F fn) -> auto {
  return fn(type_t<Ts>{}...);
}

void test1() {
  constexpr auto x = apply(pop_front(list_t<char, char>{}),
                            []<typename... Us>(type_t<Us>...) { return 42; });
  static_assert(x == 42);
}
// Reduced 1 ========================================
consteval bool zero() { return false; }

template <typename F>
consteval bool foo(bool, F f) {
  return f();
}

void test2() {
  constexpr auto x = foo(zero(), []() { return true; });
  static_assert(x);
}

// Reduced 2 ========================================
template <typename F>
consteval auto bar(F f) { return f;}

void test3() {
  constexpr auto t1 = bar(bar(bar(bar([]() { return true; }))))();
  static_assert(t1);

  int a = 1; // expected-note {{declared here}}
  auto t2 = bar(bar(bar(bar([=]() { return a; }))))(); // expected-error-re {{call to consteval function 'GH51695::bar<(lambda at {{.*}})>' is not a constant expression}}
  // expected-note@-1 {{read of non-const variable 'a' is not allowed in a constant expression}}

  constexpr auto t3 = bar(bar([x=bar(42)]() { return x; }))();
  static_assert(t3==42);
  constexpr auto t4 = bar(bar([x=bar(42)]() consteval { return x; }))();
  static_assert(t4==42);
}

}  // namespace GH51695

// https://github.com/llvm/llvm-project/issues/50455
namespace GH50455 {
void f() {
  []() consteval { int i{}; }();
  []() consteval { int i{}; ++i; }();
}
void g() {
  (void)[](int i) consteval { return i; }(0);
  (void)[](int i) consteval { return i; }(0);
}
}  // namespace GH50455

namespace GH58302 {
struct A {
   consteval A(){}
   consteval operator int() { return 1;}
};

int f() {
   int x = A{};
}
}

namespace GH57682 {
void test() {
  constexpr auto l1 = []() consteval { // expected-error {{cannot take address of consteval call operator of '(lambda at}} \
                                       // expected-note  2{{declared here}}
        return 3;
  };
  constexpr int (*f1)(void) = l1; // expected-error {{constexpr variable 'f1' must be initialized by a constant expression}} \
                                  // expected-note  {{pointer to a consteval declaration is not a constant expression}}


  constexpr auto lstatic = []() static consteval { // expected-error {{cannot take address of consteval call operator of '(lambda at}} \
                                       // expected-note  2{{declared here}} \
                                       // expected-warning {{extension}}
        return 3;
  };
  constexpr int (*f2)(void) = lstatic; // expected-error {{constexpr variable 'f2' must be initialized by a constant expression}} \
                                       // expected-note  {{pointer to a consteval declaration is not a constant expression}}

  int (*f3)(void) = []() consteval { return 3; };  // expected-error {{cannot take address of consteval call operator of '(lambda at}} \
                                                   // expected-note {{declared here}}
}

consteval void consteval_test() {
  constexpr auto l1 = []() consteval { return 3; };

  int (*f1)(void) = l1;  // ok
}
}

namespace GH60286 {

struct A {
  int i = 0;

  consteval A() {}
  A(const A&) { i = 1; }
  consteval int f() { return i; }
};

constexpr auto B = A{A{}}.f();
static_assert(B == 0);

}

namespace GH58207 {
struct tester {
    consteval tester(const char* name) noexcept { }
};
consteval const char* make_name(const char* name) { return name;}
consteval const char* pad(int P) { return "thestring"; }

int bad = 10; // expected-note 6{{declared here}}

tester glob1(make_name("glob1"));
tester glob2(make_name("glob2"));
constexpr tester cglob(make_name("cglob"));
tester paddedglob(make_name(pad(bad))); // expected-error {{call to consteval function 'GH58207::tester::tester' is not a constant expression}} \
                                        // expected-note {{read of non-const variable 'bad' is not allowed in a constant expression}}

constexpr tester glob3 = { make_name("glob3") };
constexpr tester glob4 = { make_name(pad(bad)) }; // expected-error {{call to consteval function 'GH58207::tester::tester' is not a constant expression}} \
                                                  // expected-error {{constexpr variable 'glob4' must be initialized by a constant expression}} \
                                                  // expected-note 2{{read of non-const variable 'bad' is not allowed in a constant expression}}

auto V = make_name(pad(3));
auto V1 = make_name(pad(bad)); // expected-error {{call to consteval function 'GH58207::make_name' is not a constant expression}} \
                               // expected-note {{read of non-const variable 'bad' is not allowed in a constant expression}}


void foo() {
  static tester loc1(make_name("loc1"));
  static constexpr tester loc2(make_name("loc2"));
  static tester paddedloc(make_name(pad(bad))); // expected-error {{call to consteval function 'GH58207::tester::tester' is not a constant expression}} \
                                                // expected-note {{read of non-const variable 'bad' is not allowed in a constant expression}}
}

void bar() {
  static tester paddedloc(make_name(pad(bad))); // expected-error {{call to consteval function 'GH58207::tester::tester' is not a constant expression}} \
                                                // expected-note {{read of non-const variable 'bad' is not allowed in a constant expression}}
}
}

namespace GH64949 {
struct f {
  int g; // expected-note 2{{subobject declared here}}
  constexpr ~f() {}
};
class h {

public:
  consteval h(char *) {}
  consteval operator int() const { return 1; }
  f i;
};

void test() { (int)h{nullptr}; }
// expected-error@-1 {{call to consteval function 'GH64949::h::h' is not a constant expression}}
// expected-note@-2 {{subobject 'g' is not initialized}}

int  test2() { return h{nullptr}; }
// expected-error@-1 {{call to consteval function 'GH64949::h::h' is not a constant expression}}
// expected-note@-2 {{subobject 'g' is not initialized}}


}

namespace GH65985 {

int consteval operator""_foo(unsigned long long V) {
    return 0;
}
int consteval operator""_bar(unsigned long long V); // expected-note 3{{here}}

int consteval f() {
  return 0;
}

int consteval g();  // expected-note {{here}}


struct C {
    static const int a = 1_foo;
    static constexpr int b = 1_foo;
    static const int c = 1_bar; // expected-error {{call to consteval function 'GH65985::operator""_bar' is not a constant expression}} \
                                // expected-note {{undefined function 'operator""_bar' cannot be used in a constant expression}} \
                                // expected-error {{in-class initializer for static data member is not a constant expression}}

    // FIXME: remove duplicate diagnostics
    static constexpr int d = 1_bar; // expected-error {{call to consteval function 'GH65985::operator""_bar' is not a constant expression}} \
                                    // expected-note {{undefined function 'operator""_bar' cannot be used in a constant expression}} \
                                    // expected-error {{constexpr variable 'd' must be initialized by a constant expression}}  \
                                    // expected-note {{undefined function 'operator""_bar' cannot be used in a constant expression}}

    static const int e = f();
    static const int f = g(); // expected-error {{call to consteval function 'GH65985::g' is not a constant expression}} \
                              // expected-error {{in-class initializer for static data member is not a constant expression}} \
                              // expected-note  {{undefined function 'g' cannot be used in a constant expression}}
};

}

namespace GH66562 {

namespace ns
{
    consteval int foo(int x) { return 1; } // expected-note {{declared here}}  \
                                           // expected-note {{passing argument to parameter 'x' here}}
}

template <class A>
struct T {
    static constexpr auto xx = ns::foo(A{}); // expected-error {{cannot take address of consteval function 'foo' outside of an immediate invocation}} \
                                             // expected-error {{cannot initialize a parameter of type 'int' with an rvalue of type 'char *'}}
};

template class T<char*>; // expected-note {{in instantiation}}

}

namespace GH65520 {

consteval int bar (int i) { if (i != 1) return 1/0; return 0; }
// expected-note@-1{{division by zero}}

void
g ()
{
  int a_ok[bar(1)];
  int a_err[bar(3)]; // expected-error {{call to consteval function 'GH65520::bar' is not a constant expression}} \
                     // expected-note {{in call to 'bar(3)'}}
}

consteval int undefined(); // expected-note {{declared here}}

consteval void immediate() {
    int a [undefined()]; // expected-note  {{undefined function 'undefined' cannot be used in a constant expression}} \
                         // expected-error {{call to consteval function 'GH65520::undefined' is not a constant expression}} \
                         // expected-error {{variable of non-literal type 'int[undefined()]' cannot be defined in a constexpr function before C++23}}
}


}
