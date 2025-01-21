// RUN: %clang_cc1 -std=c++20 -pedantic -verify %s

struct X {
  using type = int;
  static constexpr int value = 1;
  class tclass {};
};

template <typename T>
void f() {
  // it is a qualified name in a type-id-only context (see below), or
  // [its smallest enclosing [/new/defining/]-type-id is]:
  // - a new-type-id
  auto *Ptr = new T::type();
  // - a defining-type-id
  class T::tclass Empty1;
  T::tclass Empty2; // expected-error{{missing 'typename'}}
  // - a trailing-return-type
  auto f()->T::type;
  // - default argument of a type-parameter of a template [see below]

  // - type-id of a
  // static_cast,
  auto StaticCast = static_cast<T::type>(1.2);
  // const_cast,
  const auto *ConstCast = const_cast<const T::type *>(Ptr);
  // reinterpret_cast,
  int ReinterpretCast = reinterpret_cast<T::type>(4);
  // dynamic_cast
  struct B {
    virtual ~B() = default;
  };
  struct D : T::tclass {};
  auto *Base = dynamic_cast<T::tclass *>(new B);

  T::type Invalid; // expected-error{{missing 'typename'}}
}

template void f<X>();

// As default argument.
template <typename T, typename = T::type>
struct DefaultArg {};

template struct DefaultArg<X>;

// it is a decl-specifier of the decl-specifier-seq of a
// - simple-declaration or a function-definition in namespace scope
template <typename T>
T::type VarTemp = 1;

template int VarTemp<X>;

template <typename T>
T::type FuncDef() { return 1; }

template int FuncDef<X>();

template <typename T>
T::type funcDecl();

template <typename T>
void FuncParam(T::type); // ok, but variable template
// expected-error@-1{{variable has incomplete type 'void'}}

template <typename T>
void FuncParam2(const T::type, int); // expected-error{{missing 'typename'}}

template <typename T>
struct MemberDecl {
  // member-declaration,
  T::type Member;

  // parameter-declaration in a member-declaration, unless that
  // parameter-declaration appears in a default argument
  void NoDefault(T::type);
  void Default(int A = T::value);
};

template struct MemberDecl<X>;

// parameter-declaration in a declarator of a function or function template
// declaration where the declarator-id is qualified, unless that
// parameter-declaration appears in a default argument,
struct QualifiedFunc {
  template <typename T>
  void foo(typename T::type);
  template <typename T>
  void bar(T::type);
};

template <typename T>
void QualifiedFunc::foo(T::type) {}
template <typename T>
void QualifiedFunc::bar(typename T::type) {}

template <typename T>
void g() {
  // parameter-declaration in a lambda-declarator, unless that
  // parameter-declaration appears in a default argument, or
  auto Lambda1 = [](T::type) {};
  auto Lambda2 = [](int A = T::value) {};
}

template void g<X>();

// parameter-declaration of a (non-type) template-parameter.
template <typename T, T::type>
void NonTypeArg() {}

template void NonTypeArg<X, 0>();

template <typename T>
void f(T::type) {} // expected-error {{missing 'typename'}}

namespace N {
  template <typename T>
  int f(typename T::type);
  template <typename T>
  extern int Var;
}

template <typename T>
int N::f(T::type); // ok, function
template <typename T>
int N::Var(T::value); // ok, variable

int h() {
  return N::f<X>(10) + N::Var<X>;
}

namespace NN {
  inline namespace A { template <typename T> int f(typename T::type); } // expected-note{{previous definition is here}}
  inline namespace B { template <typename T> int f(T::type); }
}

template <typename T>
int NN::f(T::type); // expected-error{{redefinition of 'f' as different kind of symbol}}

template <auto V>
struct videntity {
  static constexpr auto value = V;
};

template <typename T,
    bool = T::value,
    bool = bool(T::value),
    bool = videntity<bool(T::value)>::value>
void f(int = T::value) {}

template <typename> int test() = delete;
template <auto> int test();

template <typename T>
int Test = test<int(T::value)>();
template int Test<X>;

template<typename T> struct A {
  enum E : T::type {}; // expected-error{{missing 'typename'}}
  operator T::type() {} // expected-error{{missing 'typename'}}
  void f() { this->operator T::type(); } // expected-error{{missing 'typename'}}
};

template<typename T>
struct C {
  C(T::type); // implicit typename context
  friend C (T::fn)(); // not implicit typename context, declarator-id of friend declaration
  C(T::type::*x)[3]; // not implicit typename context, pointer-to-member type
};

template <typename T>
C<T>::C(T::type) {}

namespace GH63119 {
struct X {
    X(int);
    X(auto);
    void f(int);
};
template<typename T> struct S {
  friend X::X(T::type);
  friend X::X(T::type = (int)(void(*)(typename T::type))(nullptr)); // expected-error {{friend declaration specifying a default argument must be a definition}}
  friend X::X(T::type = (int)(void(*)(T::type))(nullptr)); // expected-error {{friend declaration specifying a default argument must be a definition}} \
                                                           // expected-error {{expected expression}}
  friend void X::f(T::type);
};
}

namespace GH113324 {
template <typename = int> struct S1 {
  friend void f1(S1, int = 0); // expected-error {{friend declaration specifying a default argument must be a definition}}
  friend void f2(S1 a, S1 = decltype(a){}); // expected-error {{friend declaration specifying a default argument must be a definition}}
};

template <class T> using alias = int;
template <typename T> struct S2 {
  // FIXME: We miss diagnosing the default argument instantiation failure
  // (forming reference to void)
  friend void f3(S2, int a = alias<T &>(1)); // expected-error {{friend declaration specifying a default argument must be a definition}}
};

void test() {
  f1(S1<>{});
  f2(S1<>{});
  f3(S2<void>());
}
} // namespace GH113324
