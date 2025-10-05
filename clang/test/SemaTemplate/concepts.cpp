// RUN: %clang_cc1 -std=c++20 -ferror-limit 0 -verify %s

namespace PR47043 {
  template<typename T> concept True = true;
  template<typename ...T> concept AllTrue1 = True<T>; // expected-error {{expression contains unexpanded parameter pack 'T'}}
  template<typename ...T> concept AllTrue2 = (True<T> && ...);
  template<typename ...T> concept AllTrue3 = (bool)(True<T> & ...);
  static_assert(AllTrue2<int, float, char>);
  static_assert(AllTrue3<int, float, char>);
}

namespace PR47025 {
  template<typename ...T> concept AllAddable1 = requires(T ...t) { (void(t + 1), ...); };
  template<typename ...T> concept AllAddable2 = (requires(T ...t) { (t + 1); } && ...); // expected-error {{requirement contains unexpanded parameter pack 't'}}
  template<typename ...T> concept AllAddable3 = (requires(T t) { (t + 1); } && ...);
  template<typename ...T> concept AllAddable4 = requires(T t) { (t + 1); }; // expected-error {{expression contains unexpanded parameter pack 'T'}}
  template<typename ...T> concept AllAddable5 = requires(T t) { (void(t + 1), ...); }; // expected-error {{does not contain any unexpanded}}
  template<typename ...T> concept AllAddable6 = (requires { (T() + 1); } && ...);
  template<typename ...T> concept AllAddable7 = requires { (T() + 1); }; // expected-error {{expression contains unexpanded parameter pack 'T'}}

  static_assert(AllAddable1<int, float>);
  static_assert(AllAddable3<int, float>);
  static_assert(AllAddable6<int, float>);
  static_assert(!AllAddable1<int, void>);
  static_assert(!AllAddable3<int, void>);
  static_assert(!AllAddable6<int, void>);
}

namespace PR45699 {
  template<class> concept C = true; // expected-note 2{{here}}
  template<class ...Ts> void f1a() requires C<Ts>; // expected-error {{requires clause contains unexpanded parameter pack 'Ts'}}
  template<class ...Ts> requires C<Ts> void f1b(); // expected-error {{requires clause contains unexpanded parameter pack 'Ts'}}
  template<class ...Ts> void f2a() requires (C<Ts> && ...);
  template<class ...Ts> requires (C<Ts> && ...) void f2b();
  template<class ...Ts> void f3a() requires C<Ts...>; // expected-error {{pack expansion used as argument for non-pack parameter of concept}}
  template<class ...Ts> requires C<Ts...> void f3b(); // expected-error {{pack expansion used as argument for non-pack parameter of concept}}
  template<class ...Ts> void f4() {
    ([] () requires C<Ts> {} ()); // expected-error {{expression contains unexpanded parameter pack 'Ts'}}
    ([]<int = 0> requires C<Ts> () {} ()); // expected-error {{expression contains unexpanded parameter pack 'Ts'}}
  }
  template<class ...Ts> void f5() {
    ([] () requires C<Ts> {} (), ...);
    ([]<int = 0> requires C<Ts> () {} (), ...);
  }
  void g() {
    f1a();
    f1b(); // FIXME: Bad error recovery. expected-error {{undeclared identifier}}
    f2a();
    f2b();
    f3a();
    f3b(); // FIXME: Bad error recovery. expected-error {{undeclared identifier}}
    f4();
    f5();
  }
}

namespace P0857R0 {
  template <typename T> static constexpr bool V = true;

  void f() {
    auto x = []<bool B> requires B {}; // expected-note {{constraints not satisfied}} expected-note {{false}}
    x.operator()<true>();
    x.operator()<false>(); // expected-error {{no matching member function}}

    auto y = []<typename T> requires V<T> () {};
    y.operator()<int>(); // OK
  }

  template<typename T> concept C = true;
  template<template<typename T> requires C<T> typename U> struct X {};
  template<typename T> requires C<T> struct Y {};
  X<Y> xy;
}

namespace PR50306 {
  template<typename T> concept NotInt = sizeof(T) != sizeof(int); // expected-note {{because}}
  template<typename T> void f() {
    [](NotInt auto) {}(T()); // expected-error {{no matching function}} expected-note {{constraints not satisfied}} expected-note {{because}}
  }
  template void f<char>(); // OK
  template void f<int>(); // expected-note {{in instantiation of}}
}

namespace PackInTypeConstraint {
  template<typename T, typename U> concept C = sizeof(T) == sizeof(int); // expected-note 3{{}}

  template<typename ...T, C<T> U> void h1(); // expected-error {{type constraint contains unexpanded parameter pack 'T'}}
  template<typename ...T, C<T> ...U> void h2();
  template<typename ...T> void h3(C<T> auto); // expected-error {{type constraint contains unexpanded parameter pack 'T'}}
  template<typename ...T> void h4(C<T> auto...);

  template<typename ...T> void f1() {
    []<C<T> U>(U u){}(T()); // expected-error {{unexpanded parameter pack 'T'}}
  }
  template<typename ...T> void f2() {
    ([]<C<T> U>(U u){}(T()), ...); // expected-error {{no match}} expected-note 2{{}}
  }
  template void f2<int, int, int>(); // OK
  template void f2<int, char, double>(); // expected-note {{in instantiation of}}
  void f3() {
    ([]<typename ...T, C<T> U>(U u){}(0), // expected-error {{type constraint contains unexpanded parameter pack 'T'}}
     ...); // expected-error {{does not contain any unexpanded}}
  }

  template<typename ...T> void g1() {
    [](C<T> auto){}(T()); // expected-error {{expression contains unexpanded parameter pack 'T'}}
  }
  template<typename ...T> void g2() {
    ([](C<T> auto){}(T()), ...); // expected-error {{no matching function}} expected-note {{constraints not satisfied}} expected-note {{because}}
  }
  template void g2<int, int, int>(); // OK
  template void g2<int, char, double>(); // expected-note {{in instantiation of}}
  void g3() {
    ([]<typename ...T>(C<T> auto){}(1), // expected-error {{type constraint contains unexpanded parameter pack 'T'}}
     ...); // expected-error {{does not contain any unexpanded}}
  }

  template<typename ...T> void g4() {
    []() -> C<T> auto{ return T(); }(); // expected-error {{expression contains unexpanded parameter pack 'T'}}
  }
  template<typename ...T> void g5() {
    ([]() -> C<T> auto{ // expected-error-re {{deduced type {{.*}} does not satisfy}} expected-note {{while substituting into a lambda}}
     return T();
     }(), ...);
  }
  template void g5<int, int, int>(); // OK
  template void g5<int, char, double>(); // expected-note {{in instantiation of}}
  void g6() {
    ([]<typename ...T>() -> C<T> auto{ // expected-error {{declaration type contains unexpanded parameter pack 'T'}}
     return T(); // expected-error {{expression contains unexpanded parameter pack 'T'}}
     }(),
     ...); // expected-error {{does not contain any unexpanded}}
  }
}

namespace BuiltinIsConstantEvaluated {
  // Check that we do all satisfaction and diagnostic checks in a constant context.
  template<typename T> concept C = __builtin_is_constant_evaluated(); // expected-warning {{always}}
  static_assert(C<int>);

  template<typename T> concept D = __builtin_is_constant_evaluated() == true; // expected-warning {{always}}
  static_assert(D<int>);

  template<typename T> concept E = __builtin_is_constant_evaluated() == true && // expected-warning {{always}}
                                   false; // expected-note {{'false' evaluated to false}}
  static_assert(E<int>); // expected-error {{failed}} expected-note {{because 'int' does not satisfy 'E'}}

  template<typename T> concept F = __builtin_is_constant_evaluated() == false; // expected-warning {{always}}
  // expected-note@-1 {{'__builtin_is_constant_evaluated() == false' (1 == 0)}}
  static_assert(F<int>); // expected-error {{failed}} expected-note {{because 'int' does not satisfy 'F'}}

  template<typename T> concept G = __builtin_is_constant_evaluated() && // expected-warning {{always}}
                                   false; // expected-note {{'false' evaluated to false}}
  static_assert(G<int>); // expected-error {{failed}} expected-note {{because 'int' does not satisfy 'G'}}
}

namespace NoConstantFolding {
  // Ensure we use strict constant evaluation rules when checking satisfaction.
  int n;
  template <class T> concept C = &n + 3 - 3 == &n; // expected-error {{non-constant expression}} expected-note {{cannot refer to element 3 of non-array object}}
  static_assert(C<void>); // expected-note {{while checking}}
}

namespace PR50337 {
  template <typename T> concept foo = true;
  template <typename T> concept foo2 = foo<T> && true;
  void f(foo auto, auto);
  void f(foo2 auto, auto);
  void g() { f(1, 2); }
}

namespace PR50561 {
  template<typename> concept C = false;
  template<typename T, typename U> void f(T, U);
  template<C T, typename U> void f(T, U) = delete;
  void g() { f(0, 0); }
}

namespace PR49188 {
  template<class T> concept C = false;     // expected-note 7 {{because 'false' evaluated to false}}

  C auto f1() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
    return void();
  }
  C auto f2() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
    return;
  }
  C auto f3() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
  }
  C decltype(auto) f4() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
    return void();
  }
  C decltype(auto) f5() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
    return;
  }
  C decltype(auto) f6() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
  }
  C auto& f7() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
    return void();
  }
  C auto& f8() {
    return; // expected-error {{cannot deduce return type 'C auto &' from omitted return expression}}
  }
  C auto& f9() { // expected-error {{cannot deduce return type 'C auto &' for function with no return statements}}
  }
}
namespace PR53911 {
  template<class T> concept C = false; // expected-note 3 {{because 'false' evaluated to false}}

  C auto *f1() { // expected-error {{deduced type 'void' does not satisfy 'C'}}
    return (void*)nullptr;
  }
  C auto *f2() { // expected-error {{deduced type 'int' does not satisfy 'C'}}
    return (int*)nullptr;
  }
  C auto *****f3() { // expected-error {{deduced type 'int' does not satisfy 'C'}}
    return (int*****)nullptr;
  }
}

namespace PR54379 {
template <int N>
struct A {
  static void f() requires (N == 0) { return; } // expected-note {{candidate template ignored: constraints not satisfied}} expected-note {{evaluated to false}}
  static void f() requires (N == 1) { return; } // expected-note {{candidate template ignored: constraints not satisfied}} expected-note {{evaluated to false}}
};
void (*f1)() = A<2>::f; // expected-error {{address of overloaded function 'f' does not match required type}}

struct B {
  template <int N2 = 1> static void f() requires (N2 == 0) { return; }  // expected-note {{candidate template ignored: constraints not satisfied [with N2 = 1]}} expected-note {{evaluated to false}}
};
void (*f2)() = B::f; // expected-error {{address of overloaded function 'f' does not match required type}}
}

namespace PR54443 {

template <class T, class U>
struct is_same { static constexpr bool value = false; };

template <class T>
struct is_same<T, T> { static constexpr bool value = true; };

template <class T, class U>
concept same_as = is_same<T, U>::value; // expected-note-re 4 {{because {{.*}} evaluated to false}}

int const &f();

same_as<int const> auto i1 = f(); // expected-error {{deduced type 'int' does not satisfy 'same_as<const int>'}}
same_as<int const> auto &i2 = f();
same_as<int const> auto &&i3 = f(); // expected-error {{deduced type 'const int &' does not satisfy 'same_as<const int>'}}

same_as<int const &> auto i4 = f(); // expected-error {{deduced type 'int' does not satisfy 'same_as<const int &>'}}
same_as<int const &> auto &i5 = f(); // expected-error {{deduced type 'const int' does not satisfy 'same_as<const int &>'}}
same_as<int const &> auto &&i6 = f();

template <class T>
concept C = false; // expected-note 3 {{because 'false' evaluated to false}}

int **const &g();

C auto **j1 = g();   // expected-error {{deduced type 'int' does not satisfy 'C'}}
C auto **&j2 = g();  // expected-error {{deduced type 'int' does not satisfy 'C'}}
C auto **&&j3 = g(); // expected-error {{deduced type 'int' does not satisfy 'C'}}
}

namespace GH55567 {
template<class, template <class> class> concept C = true;
template <class> struct S {};
void f(C<GH55567::S> auto);
} // namespace GH55567

namespace SubConstraintChecks {
template <typename T>
concept TrueConstraint = true;
template <typename T>
concept FalseConstraint = false;

template <typename T, typename... Us>
class ContainsConstrainedFuncTrue {
public:
  template <typename V, TrueConstraint Constrained>
  static void func(V &&, Constrained &&C);
};
template <typename T, typename... Us>
class ContainsConstrainedFuncFalse {
public:
  template <typename V, FalseConstraint Constrained>
  static void func(V &&, Constrained &&C);
};

template <typename... Us>
concept TrueConstraint2 =
    requires(float &&t) {
      ContainsConstrainedFuncTrue<float, Us...>::func(5, 0.0);
    };
template <typename... Us>
concept FalseConstraint2 =
    requires(float &&t) {
      ContainsConstrainedFuncFalse<float, Us...>::func(5, 0.0); // #FC2_CONSTR
    };

template <typename T>
void useTrue(int F)
  requires TrueConstraint2<int>
{}

template <typename T>
void useFalse(int F)             // #USE_FALSE
  requires FalseConstraint2<int> // #USE_FALSE_CONSTR
{}

// Should only diagnose 'false' once instantiated.
void UseUse() {
  useTrue<int>(5);
  useFalse<int>(5);
  // expected-error@-1{{no matching function for call to 'useFalse'}}
  // expected-note@#USE_FALSE{{constraints not satisfied}}
  // expected-note@#USE_FALSE_CONSTR{{because 'int' does not satisfy 'FalseConstraint2'}}
  // expected-note@#FC2_CONSTR {{would be invalid: no matching function for call to 'func'}}
}
} // namespace SubConstraintChecks

namespace DeducedTemplateArgs {
template <typename Itr> struct ItrTraits {
  template <typename PtrItr> struct Ptr {
  };
  template <typename PtrItr>
    requires requires { typename PtrItr::pointer; }
  struct Ptr<PtrItr> {
    using type = typename Itr::pointer;
  };
  using pointer = typename Ptr<Itr>::type; // #TRAITS_PTR
};

struct complete_itr {
  using pointer = int;
};

template <typename T> class Complete {
  using ItrType = ItrTraits<complete_itr>;
  ItrType begin() noexcept { return ItrType(); }
};

// This version doesn't have 'pointer', so error confirms we are in the first
// verison of 'Ptr'.
struct not_complete_itr {
};

template <typename T> class NotComplete {
  using ItrType = ItrTraits<not_complete_itr>;
  ItrType begin() noexcept { return ItrType(); }
  // expected-error@#TRAITS_PTR{{no type named 'type' in }}
  // expected-note@-2{{in instantiation of template class }}
};
} // namespace DeducedTemplateArgs

namespace DeferredInstantiationInstScope {
template <typename T>
struct remove_ref {
  using type = T;
};
template <typename T>
struct remove_ref<T &> {
  using type = T;
};
template <typename T>
struct remove_ref<T &&> {
  using type = T;
};

template <typename T>
constexpr bool IsInt = PR54443::is_same<typename remove_ref<T>::type,
                                        int>::value;

template <typename U>
void SingleDepthReferencesTop(U &&u) {
  struct lc {
    void operator()()             // #SDRT_OP
      requires IsInt<decltype(u)> // #SDRT_REQ
    {}
  };
  lc lv;
  lv(); // #SDRT_CALL
}

template <typename U>
void SingleDepthReferencesTopNotCalled(U &&u) {
  struct lc {
    void operator()()
      requires IsInt<typename decltype(u)::FOO>
    {}
  };
  lc lv;
}

template <typename U>
void SingleDepthReferencesTopCalled(U &&u) {
  struct lc {
    void operator()()                           // #CALLOP
      requires IsInt<typename decltype(u)::FOO> // #CONSTR
    {}
  };
  lc lv;
  lv();
  // expected-error@-1{{no matching function for call to object of type 'lc'}}
  // expected-note@#SDRTC{{in instantiation of function template}}
  // expected-note@#CALLOP{{constraints not satisfied}}
  // expected-note@#CONSTR{{substituted constraint expression is ill-formed}}
}

template <typename U>
void SingleDepthReferencesTopLambda(U &&u) {
  []() // #SDRTL_OP
    requires IsInt<decltype(u)> // #SDRTL_REQ
  {}();
}

template <typename U>
void DoubleDepthReferencesTop(U &&u) {
  struct lc { // #DDRT_STRCT
    void operator()() {
      struct lc2 {
        void operator()()             // #DDRT_OP
          requires IsInt<decltype(u)> // #DDRT_REQ
        {}
      };
      lc2 lv2;
      lv2(); // #DDRT_CALL
    }
  };
  lc lv;
  lv();
}

template <typename U>
void DoubleDepthReferencesTopLambda(U &&u) {
  []() { []() // #DDRTL_OP
           requires IsInt<decltype(u)> // #DDRTL_REQ
         {}(); }();
}

template <typename U>
void DoubleDepthReferencesAll(U &&u) {
  struct lc { // #DDRA_STRCT
    void operator()(U &&u2) {
      struct lc2 {
        void operator()(U &&u3)          // #DDRA_OP
          requires IsInt<decltype(u)> && // #DDRA_REQ
                   IsInt<decltype(u2)> && IsInt<decltype(u3)>
        {}
      };
      lc2 lv2;
      lv2(u2); // #DDRA_CALL
    }
  };
  lc lv;
  lv(u);
}

template <typename U>
void DoubleDepthReferencesAllLambda(U &&u) {
  [](U &&u2) { // #DDRAL_OP1
    [](U && u3) // #DDRAL_OP2
      requires IsInt<decltype(u)> // #DDRAL_REQ
            && IsInt<decltype(u2)>
            && IsInt<decltype(u3)>
    {}(u2);
  }(u);
}

template <typename U>
struct CausesFriendConstraint {
  template <typename V>
  friend void FriendFunc(CausesFriendConstraint, V) // #FF_DECL
    requires IsInt<U> &&
             IsInt<V> // #FF_REQ
  {}
};
// FIXME: Re-enable this test when constraints are allowed to refer to captures.
// template<typename T>
// void ChecksCapture(T x) {
//   [y = x]() requires(IsInt<decltype(y)>){}();
// }

template <typename T>
void ChecksLocalVar(T x) {
  T Local;
  []() // #CLV_OP
    requires(IsInt<decltype(Local)>) // #CLV_REQ
  {}();
}

template <typename T>
void LocalStructMemberVar(T x) {
  struct S {
    T local;
    void foo()
      requires(IsInt<decltype(local)>) // #LSMV_REQ
    {}
  } s;
  s.foo(); // #LSMV_CALL
};

template <typename T>
struct ChecksMemberVar {
  T t;
  void foo()
    requires(IsInt<decltype(t)>) // #CMV_FOO
  {}
  template <typename U>
  void foo2()                    // #CMV_FOO2
    requires(IsInt<decltype(t)>) // #CMV_FOO2_REQ
  {}
};

void test_dependent() {
  int v = 0;
  float will_fail;
  SingleDepthReferencesTop(v);
  SingleDepthReferencesTop(will_fail);
  // expected-error@#SDRT_CALL{{no matching function for call to object of type 'lc'}}
  // expected-note@-2{{in instantiation of function template specialization}}
  // expected-note@#SDRT_OP{{candidate function not viable}}
  // expected-note@#SDRT_REQ{{'IsInt<decltype(u)>' evaluated to false}}

  SingleDepthReferencesTopNotCalled(v);
  // Won't error unless we try to call it.
  SingleDepthReferencesTopNotCalled(will_fail);
  SingleDepthReferencesTopCalled(v); // #SDRTC
  SingleDepthReferencesTopLambda(v);
  SingleDepthReferencesTopLambda(will_fail);
  // expected-note@-1{{in instantiation of function template specialization}}
  // expected-error@#SDRTL_OP{{no matching function for call to object of type}}
  // expected-note@#SDRTL_OP{{candidate function not viable: constraints not satisfied}}
  // expected-note@#SDRTL_REQ{{because 'IsInt<decltype(u)>' evaluated to false}}

  DoubleDepthReferencesTop(v);
  DoubleDepthReferencesTop(will_fail);
  // expected-error@#DDRT_CALL{{no matching function for call to object of type 'lc2'}}
  // expected-note@-2{{in instantiation of function template specialization}}
  // expected-note@#DDRT_STRCT{{in instantiation of member function}}
  // expected-note@#DDRT_OP{{candidate function not viable}}
  // expected-note@#DDRT_REQ{{'IsInt<decltype(u)>' evaluated to false}}

  DoubleDepthReferencesTopLambda(v);
  DoubleDepthReferencesTopLambda(will_fail);
  // expected-note@-1{{in instantiation of function template specialization}}
  // expected-error@#DDRTL_OP{{no matching function for call to object of type}}
  // expected-note@#DDRTL_OP{{candidate function not viable: constraints not satisfied}}
  // expected-note@#DDRTL_OP{{while substituting into a lambda expression here}}
  // expected-note@#DDRTL_REQ{{because 'IsInt<decltype(u)>' evaluated to false}}
  DoubleDepthReferencesAll(v);
  DoubleDepthReferencesAll(will_fail);
  // expected-error@#DDRA_CALL{{no matching function for call to object of type 'lc2'}}
  // expected-note@-2{{in instantiation of function template specialization}}
  // expected-note@#DDRA_STRCT{{in instantiation of member function}}
  // expected-note@#DDRA_OP{{candidate function not viable}}
  // expected-note@#DDRA_REQ{{'IsInt<decltype(u)>' evaluated to false}}

  DoubleDepthReferencesAllLambda(v);
  DoubleDepthReferencesAllLambda(will_fail);
  // expected-note@-1{{in instantiation of function template specialization}}
  // expected-note@#DDRAL_OP1{{while substituting into a lambda expression here}}
  // expected-error@#DDRAL_OP2{{no matching function for call to object of type}}
  // expected-note@#DDRAL_OP2{{candidate function not viable: constraints not satisfied}}
  // expected-note@#DDRAL_REQ{{because 'IsInt<decltype(u)>' evaluated to false}}

  CausesFriendConstraint<int> CFC;
  FriendFunc(CFC, 1);
  FriendFunc(CFC, 1.0);
  // expected-error@-1{{no matching function for call to 'FriendFunc'}}
  // expected-note@#FF_DECL{{constraints not satisfied}}
  // expected-note@#FF_REQ{{because 'IsInt<double>' evaluated to false}}

  // FIXME: Re-enable this test when constraints are allowed to refer to captures.
  // ChecksCapture(v);

  ChecksLocalVar(v);
  ChecksLocalVar(will_fail);
  // expected-note@-1{{in instantiation of function template specialization}}
  // expected-error@#CLV_OP{{no matching function for call to object of type}}
  // expected-note@#CLV_OP{{candidate function not viable: constraints not satisfied}}
  // expected-note@#CLV_REQ{{because 'IsInt<decltype(Local)>' evaluated to false}}



  LocalStructMemberVar(v);
  LocalStructMemberVar(will_fail);
  // expected-error@#LSMV_CALL{{invalid reference to function 'foo'}}
  // expected-note@-2{{in instantiation of function template specialization}}
  // expected-note@#LSMV_REQ{{because 'IsInt<decltype(this->local)>' evaluated to false}}

  ChecksMemberVar<int> CMV;
  CMV.foo();
  CMV.foo2<int>();

  ChecksMemberVar<float> CMV2;
  CMV2.foo();
  // expected-error@-1{{invalid reference to function 'foo'}}
  // expected-note@#CMV_FOO{{because 'IsInt<decltype(this->t)>' evaluated to false}}
  CMV2.foo2<float>();
  // expected-error@-1{{no matching member function for call to 'foo2'}}
  // expected-note@#CMV_FOO2{{constraints not satisfied}}
  // expected-note@#CMV_FOO2_REQ{{because 'IsInt<decltype(this->t)>' evaluated to false}}
}
} // namespace DeferredInstantiationInstScope

// Ane example of evaluating a concept at two different depths in the same
// evaluation.  No diagnostic is expected.
namespace SameConceptDifferentDepth {
template <class _Ip>
concept sentinel_for =
    requires(_Ip __i) {
      __i++;
    };

template <class _Ip>
concept bidirectional_iterator =
    sentinel_for<_Ip>;

template <class _Iter>
class move_iterator {
public:
  auto operator++(int)
    requires sentinel_for<_Iter>{}
};

static_assert(bidirectional_iterator<move_iterator<int>>);
} // namespace SameConceptDifferentDepth

namespace VarInit {
template <class _Tp>
concept __can_reference = true;

template <class _Iter>
class common_iterator {
public:
  common_iterator() {
    constexpr auto x = requires(_Iter & __i) { { __i } -> __can_reference; };
  }
};

void test() {
  auto commonIter1 = common_iterator<int>();
}
} // namespace VarInit


namespace InlineFriendOperator {
template <typename T>
concept C = true;
template <class _Iter>
class counted_iterator {
  _Iter I;
public:
  constexpr counted_iterator() = default;
  friend constexpr auto operator+( // expected-note {{candidate function not viable}}
      int __n, const counted_iterator &__x)
    requires C<decltype(I)>
  {
    return __x + __n; // expected-error{{invalid operands to binary expression}}
  }
};

constexpr bool test() {
  counted_iterator<int> iter;
  auto x = 2 + iter; // expected-note{{in instantiation of member function 'InlineFriendOperator::operator+'}}

  return true;
}
} // namespace InlineFriendOperator

namespace ClassTemplateInstantiation {
struct Type;
template < typename A, typename B, typename C>
  concept ConstraintF = false; // #ConstraintF
template < typename A, typename B, typename C>
  concept ConstraintT = true;

template < typename T > struct Parent {
  template < typename U, ConstraintT<T, U> > struct ChildT{};
  ChildT<Type, Type> CauseInstT;
  template < typename U, ConstraintF<T, U> > struct ChildF{};// #ChildF
  ChildF<Type, Type> CauseInstF; //#CauseInstF
};

// expected-error@#CauseInstF{{constraints not satisfied for class template}}
// expected-note@+3{{in instantiation of template class}}
// expected-note@#ChildF{{evaluated to false}}
// expected-note@#ConstraintF{{because 'false' evaluated to false}}
Parent<int> Inst;
} // namespace ClassTemplateInstantiation

namespace SelfFriend {
  template<class T>
  concept Constraint = requires (T i) { (*i); };
  template<class T>
  concept Constraint2 = requires (T i) { (*i); };

  template<Constraint T>
  struct Iterator {
    template <Constraint>
    friend class Iterator;
    void operator*();
  };

  template<Constraint T> // #ITER_BAD
  struct IteratorBad {
    template <Constraint2>//#ITER_BAD_FRIEND
    friend class IteratorBad;
    void operator*();
  };

  Iterator<int*> I;
  Iterator<char*> I2;
  IteratorBad<int*> I3; // expected-error@#ITER_BAD_FRIEND{{constraint differs}}
                        // expected-note@-1{{in instantiation of template class}}
                        // expected-note@#ITER_BAD{{previous template declaration}}
} // namespace SelfFriend


namespace Surrogates {
int f1(int);
template <auto N>
struct A {
    using F = int(int);
    operator F*() requires N { return f1; } // expected-note{{conversion candidate 'operator int (*)(int)' not viable: constraints not satisfied}}
};
int i = A<true>{}(0);
int j = A<false>{}(0); // expected-error{{no matching function for call to object of type 'A<false>'}}
}


namespace ConstrainedMemberVarTemplate {
template <long Size> struct Container {
  static constexpr long arity = Size;
  template <typename U>
  requires(sizeof(U) == arity) // #CMVT_REQ
  using var_templ = int;
};
Container<4>::var_templ<int> inst;
Container<5>::var_templ<int> inst_fail;
// expected-error@-1{{constraints not satisfied for alias template 'var_templ'}}
// expected-note@#CMVT_REQ{{because 'sizeof(int) == arity' (4 == 5) evaluated to false}}
} // namespace ConstrainedMemberVarTemplate

// These should not diagnose, where we were unintentionally doing so before by
// checking trailing requires clause twice, yet not having the ability to the
// 2nd time, since it was no longer a dependent variant.
namespace InheritedFromPartialSpec {
template<class C>
constexpr bool Check = true;

template<typename T>
struct Foo {
  template<typename U>
    Foo(U&&) requires (Check<U>){}
  template<typename U>
    void MemFunc(U&&) requires (Check<U>){}
  template<typename U>
    static void StaticMemFunc(U&&) requires (Check<U>){}
  ~Foo() requires (Check<T>){}
};

template<>
  struct Foo<void> : Foo<int> {
    using Foo<int>::Foo;
    using Foo<int>::MemFunc;
    using Foo<int>::StaticMemFunc;
  };

void use() {
  Foo<void> F {1.1};
  F.MemFunc(1.1);
  Foo<void>::StaticMemFunc(1.1);
}

template<typename T>
struct counted_iterator {
  constexpr auto operator->() const noexcept requires false {
    return T::Invalid;
  };
};

template<class _Ip>
concept __has_member_pointer = requires { typename _Ip::pointer; };

template<class>
struct __iterator_traits_member_pointer_or_arrow_or_void { using type = void; };
template<__has_member_pointer _Ip>
struct __iterator_traits_member_pointer_or_arrow_or_void<_Ip> { using type = typename _Ip::pointer; };

template<class _Ip>
  requires requires(_Ip& __i) { __i.operator->(); } && (!__has_member_pointer<_Ip>)
struct __iterator_traits_member_pointer_or_arrow_or_void<_Ip> {
  using type = decltype(declval<_Ip&>().operator->());
};


void use2() {
  __iterator_traits_member_pointer_or_arrow_or_void<counted_iterator<int>> f;
}
}// namespace InheritedFromPartialSpec

namespace GH48182 {
template<typename, typename..., typename = int> // expected-error{{template parameter pack must be the last template parameter}}
concept invalid = true;

template<typename> requires invalid<int> // expected-error{{use of undeclared identifier 'invalid'}}
no errors are printed
;

static_assert(invalid<int> also here ; // expected-error{{use of undeclared identifier 'invalid'}}

int foo() {
    bool b;
    b = invalid<int> not just in declarations; // expected-error{{use of undeclared identifier 'invalid'}}
    return b;
}
} // namespace GH48182

namespace GH61777 {
template<class T> concept C = sizeof(T) == 4; // #61777_C
template<class T, class U> concept C2 = sizeof(T) == sizeof(U); //#61777_C2

template<class T>
struct Parent {
  template<class, C auto> struct TakesUnary { static const int i = 0 ; }; // #UNARY
  template<class, C2<T> auto> struct TakesBinary { static const int i = 0 ; }; //#BINARY
};

static_assert(Parent<void>::TakesUnary<int, 0>::i == 0);
// expected-error@+3{{constraints not satisfied for class template 'TakesUnary'}}
// expected-note@#UNARY{{because 'decltype(0ULL)' (aka 'unsigned long long') does not satisfy 'C'}}
// expected-note@#61777_C{{because 'sizeof(unsigned long long) == 4' (8 == 4) evaluated to false}}
static_assert(Parent<void>::TakesUnary<int, 0uLL>::i == 0);

static_assert(Parent<int>::TakesBinary<int, 0>::i == 0);
// expected-error@+3{{constraints not satisfied for class template 'TakesBinary'}}
// expected-note@#BINARY{{because 'C2<decltype(0ULL), int>' evaluated to false}}
// expected-note@#61777_C2{{because 'sizeof(unsigned long long) == sizeof(int)' (8 == 4) evaluated to false}}
static_assert(Parent<int>::TakesBinary<int, 0ULL>::i == 0);
}

namespace TemplateInsideNonTemplateClass {
template<typename T, typename U> concept C = true;

template<typename T> auto L = []<C<T> U>() {};

struct Q {
  template<C<int> U> friend constexpr auto decltype(L<int>)::operator()() const;
};

template <class T>
concept C1 = false;

struct Foo {
  template <typename>
  struct Bar {};

  template <typename T>
    requires(C1<T>)
  struct Bar<T>;
};

Foo::Bar<int> BarInstance;
} // namespace TemplateInsideNonTemplateClass

namespace GH61959 {
template <typename T0>
concept C = (sizeof(T0) >= 4);

template<typename...>
struct Orig { };

template<typename T>
struct Orig<T> {
  template<typename> requires C<T>
  void f() { }

  template<typename> requires true
  void f() { }
};

template <typename...> struct Mod {};

template <typename T1, typename T2>
struct Mod<T1, T2> {
  template <typename> requires C<T1>
  constexpr static int f() { return 1; }

  template <typename> requires C<T2>
  constexpr static int f() { return 2; }
};

static_assert(Mod<int, char>::f<double>() == 1);
static_assert(Mod<char, int>::f<double>() == 2);

template<typename T>
struct Outer {
  template<typename ...>
  struct Inner {};

  template<typename U>
  struct Inner<U> {
    template<typename V>
    void foo()  requires C<U> && C<T> && C<V>{}
    template<typename V>
    void foo()  requires true{}
  };
};

void bar() {
  Outer<int>::Inner<float> I;
  I.foo<char>();
}
} // namespace GH61959


namespace TemplateInsideTemplateInsideTemplate {
template<typename T>
concept C1 = false;

template <unsigned I0>
struct W0 {
  template <unsigned I1>
  struct W1 {
    template <typename T>
    struct F {
      enum { value = 1 };
    };

    template <typename T>
      requires C1<T>
    struct F<T> {
      enum { value = 2 };
    };
  };
};

static_assert(W0<0>::W1<1>::F<int>::value == 1);
} // TemplateInsideTemplateInsideTemplate


namespace GH63181 {

template<auto N, class T> void f() {
auto l = []() requires N { }; // expected-note 2{{candidate function not viable: constraints not satisfied}} \
                              // expected-note 2{{because 'false' evaluated to false}}

l();
// expected-error@-1 {{no matching function for call to object of type}}
void(*ptr)() = l;
// expected-error-re@-1 {{no viable conversion from '(lambda {{.*}})' to 'void (*)()'}}
}

template void f<false, int>(); // expected-note {{in instantiation of function template specialization 'GH63181::f<false, int>' requested here}}
template void f<true, int>();

template<class T> concept C = __is_same(T, int); // expected-note{{because '__is_same(char, int)' evaluated to false}}

template<class... Ts> void f() {
  ([]() requires C<Ts> { return Ts(); }(), ...);
  // expected-error@-1 {{no matching function for call to object of type}} \
  // expected-note@-1 {{candidate function not viable: constraints not satisfied}} \
  // expected-note@-1 {{because 'char' does not satisfy 'C'}}
}

template void f<int, int, int>();
template void f<int, int, char>();
//expected-note@-1{{in instantiation of function template specialization 'GH63181::f<int, int, char>' requested here}}


template <typename T, bool IsTrue>
concept Test = IsTrue; // expected-note 2{{because 'false' evaluated to false}}

template <typename T, bool IsTrue>
void params() {
    auto l = [](T t)  // expected-note 2{{candidate function not viable: constraints not satisfied}}
    requires Test<decltype(t), IsTrue> // expected-note 2{{because 'Test<decltype(t), false>' evaluated to false}}
    {};
    using F = void(T);
    F* f = l; // expected-error {{no viable conversion from}}
    l(0); // expected-error {{no matching function for call to object}}
}

void test_params() {
    params<int, true>();
    params<int, false>(); // expected-note {{in instantiation of function template specialization 'GH63181::params<int, false>' requested here}}
}

}

namespace GH54678 {
template<class>
concept True = true;

template<class>
concept False = false; // expected-note 9 {{'false' evaluated to false}}

template<class>
concept Irrelevant = false;

template <typename T>
concept ErrorRequires = requires(ErrorRequires auto x) { x; }; //#GH54678-ill-formed-concept
// expected-error@-1 {{a concept definition cannot refer to itself}} \
// expected-error@-1 {{'auto' not allowed in requires expression parameter}} \
// expected-note@-1 {{declared here}}

template<typename T> concept C1 = C1<T> && []<C1>(C1 auto) -> C1 auto {};
//expected-error@-1 4{{a concept definition cannot refer to itself}} \
//expected-note@-1 4{{declared here}}

template<class T> void aaa(T t) // expected-note {{candidate template ignored: constraints not satisfied}}
requires (False<T> || False<T>) || False<T> {} // expected-note 3 {{'int' does not satisfy 'False'}}
template<class T> void bbb(T t) // expected-note {{candidate template ignored: constraints not satisfied}}
requires (False<T> || False<T>) && True<T> {} // expected-note 2 {{'long' does not satisfy 'False'}}
template<class T> void ccc(T t) // expected-note {{candidate template ignored: constraints not satisfied}}
requires (True<T> || Irrelevant<T>) && False<T> {} // expected-note {{'unsigned long' does not satisfy 'False'}}
template<class T> void ddd(T t) // expected-note {{candidate template ignored: constraints not satisfied}}
requires (Irrelevant<T> || True<T>) && False<T> {} // expected-note {{'int' does not satisfy 'False'}}
template<class T> void eee(T t) // expected-note {{candidate template ignored: constraints not satisfied}}
requires (Irrelevant<T> || Irrelevant<T> || True<T>) && False<T> {} // expected-note {{'long' does not satisfy 'False'}}

template<class T> void fff(T t) // expected-note {{candidate template ignored: constraints not satisfied}}
requires((ErrorRequires<T> || False<T> || True<T>) && False<T>) {} // expected-note {{because 'unsigned long' does not satisfy 'False'}}
void test() {
    aaa(42); // expected-error {{no matching function}}
    bbb(42L); // expected-error{{no matching function}}
    ccc(42UL); // expected-error {{no matching function}}
    ddd(42); // expected-error {{no matching function}}
    eee(42L); // expected-error {{no matching function}}
    fff(42UL); // expected-error {{no matching function}}
}
}

namespace GH66612 {
  template<typename C>
    auto end(C c) ->int;

  template <typename T>
    concept Iterator = true;

  template <typename CT>
    concept Container = requires(CT b) {
        { end } -> Iterator; // #66612GH_END
    };

  static_assert(Container<int>);// expected-error{{static assertion failed}}
  // expected-note@-1{{because 'int' does not satisfy 'Container'}}
  // expected-note@#66612GH_END{{because 'end' would be invalid: reference to overloaded function could not be resolved; did you mean to call it?}}
}

namespace GH66938 {
template <class>
concept True = true;

template <class>
concept False = false;

template <class T>
void cand(T t)
  requires False<T> || False<T> || False<T> || False<T> || False<T> ||
           False<T> || False<T> || False<T> || False<T> || True<T>
{}

void test() { cand(42); }
}

namespace GH63837 {

template<class> concept IsFoo = true;

template<class> struct Struct {
  template<IsFoo auto... xs>
  void foo() {}

  template<auto... xs> requires (... && IsFoo<decltype(xs)>)
  void bar() {}

  template<IsFoo auto... xs>
  static inline int field = 0;
};

template void Struct<void>::foo<>();
template void Struct<void>::bar<>();
template int Struct<void>::field<1, 2>;

}

namespace GH64808 {

template <class T> struct basic_sender {
  T func;
  basic_sender(T) : func(T()) {}
};

auto a = basic_sender{[](auto... __captures) {
  return []() // #note-a-1
    requires((__captures, ...), false) // #note-a-2
  {};
}()};

auto b = basic_sender{[](auto... __captures) {
  return []()
    requires([](int, double) { return true; }(decltype(__captures)()...))
  {};
}(1, 2.33)};

void foo() {
  a.func();
  // expected-error@-1{{no matching function for call}}
  // expected-note@#note-a-1{{constraints not satisfied}}
  // expected-note@#note-a-2{{evaluated to false}}
  b.func();
}

} // namespace GH64808

namespace GH86757_1 {
template <typename...> concept b = false;
template <typename> concept c = b<>;
template <typename d> concept f = c< d >;
template <f> struct e; // expected-note {{}}
template <f d> struct e<d>; // expected-error {{class template partial specialization is not more specialized than the primary template}}
}


namespace constrained_variadic {
template <typename T = int>
struct S {
    void f(); // expected-note {{candidate}}
    void f(...) requires true;   // expected-note {{candidate}}

    void g(...);  // expected-note {{candidate}}
    void g() requires true;  // expected-note {{candidate}}

    consteval void h(...);
    consteval void h(...) requires true {};
};

int test() {
    S{}.f(); // expected-error{{call to member function 'f' is ambiguous}}
    S{}.g(); // expected-error{{call to member function 'g' is ambiguous}}
    S{}.h();
}

}

namespace GH109780 {

template <typename T>
concept Concept; // expected-error {{expected '='}}

bool val = Concept<int>;

template <typename T>
concept C = invalid; // expected-error {{use of undeclared identifier 'invalid'}}

bool val2 = C<int>;

} // namespace GH109780

namespace GH121980 {

template <class>
concept has_member_difference_type; // expected-error {{expected '='}}

template <has_member_difference_type> struct incrementable_traits; // expected-note {{declared here}}

template <has_member_difference_type Tp>
struct incrementable_traits<Tp>; // expected-error {{not more specialized than the primary}}

}

namespace InjectedClassNameType {

template <class, class _Err> class expected {
public:
  template <class...>
  expected(...);

  template <class _T2, class _E2>
  friend bool operator==(expected x, expected<_T2, _E2>)
    requires requires {
      { x };
    }
  {
    return true;
  }
};

bool test_val_types() {
  return expected<void, int>() == 1;
}

}

namespace CWG2369_Regression {

enum class KindEnum {
  Unknown = 0,
  Foo = 1,
};

template <typename T>
concept KnownKind = T::kind() != KindEnum::Unknown;

template <KnownKind T> struct KnownType;

struct Type {
  KindEnum kind() const;

  static Type f(Type t);

  template <KnownKind T> static KnownType<T> f(T t);

  static void g() {
    Type t;
    f(t);
  }
};

template <KnownKind T> struct KnownType {
  static constexpr KindEnum kind() { return KindEnum::Foo; }
};

}

namespace CWG2369_Regression_2 {

template <typename T>
concept HasFastPropertyForAttribute =
    requires(T element, int name) { element.propertyForAttribute(name); };

template <typename OwnerType>
struct SVGPropertyOwnerRegistry {
  static int fastAnimatedPropertyLookup() {
    static_assert (HasFastPropertyForAttribute<OwnerType>);
    return 1;
  }
};

class SVGCircleElement {
  friend SVGPropertyOwnerRegistry<SVGCircleElement>;
  void propertyForAttribute(int);
};

int i = SVGPropertyOwnerRegistry<SVGCircleElement>::fastAnimatedPropertyLookup();

}

namespace GH61824 {

template<typename T, typename U = typename T::type> // #T_Type
concept C = true;

constexpr bool f(C auto) { // #GH61824_f
  return true;
}

C auto x = 0;
// expected-error@#T_Type {{type 'int' cannot be used prior to '::'}} \
// expected-note@-1 {{in instantiation of default argument}}

static_assert(f(0));

}

namespace GH149986 {
template <typename T> concept PerfectSquare = [](){} // expected-note 2{{here}}
([](auto) { return true; }) < PerfectSquare <class T>;
// expected-error@-1 {{declaration of 'T' shadows template parameter}} \
// expected-error@-1 {{a concept definition cannot refer to itself}}

}
namespace GH61811{
template <class T> struct A { static const int x = 42; };
template <class Ta> concept A42 = A<Ta>::x == 42;
template <class Tv> concept Void = __is_same_as(Tv, void);
template <class Tb, class Ub> concept A42b = Void<Tb> || A42<Ub>;
template <class Tc> concept R42c = A42b<Tc, Tc&>;
static_assert (R42c<void>);
}

namespace parameter_mapping_regressions {

namespace case1 {

template <template <class> class> using __meval = struct __q;
template <template <class> class _Tp>
concept __mvalid = requires { typename __meval<_Tp>; };
template <class _Fn>
concept __minvocable = __mvalid<_Fn::template __f>;
template <class...> struct __mdefer_;
template <class _Fn, class... _Args>
  requires __minvocable<_Fn>
struct __mdefer_<_Fn, _Args...> {};
template <class = __q> struct __mtransform {
  template <class> using __f = int;
};
struct __completion_domain_or_none_ : __mdefer_<__mtransform<>> {};

}

namespace case2 {

template<auto& Q, class P> concept C = Q.template operator()<P>();
template<class P> concept E = C<[]<class Ty>{ return false; }, P>;
static_assert(!E<int>);

}


namespace case3 {
template <class> constexpr bool is_move_constructible_v = false;

template <class _Tp>
concept __cpp17_move_constructible = is_move_constructible_v<_Tp>; // #is_move_constructible_v

template <class _Tp>
concept __cpp17_copy_constructible = __cpp17_move_constructible<_Tp>; // #__cpp17_move_constructible

template <class _Iter>
concept __cpp17_iterator = __cpp17_copy_constructible<_Iter>; // #__cpp17_copy_constructible

struct not_move_constructible {};
static_assert(__cpp17_iterator<not_move_constructible>); \
// expected-error {{static assertion failed}} \
// expected-note {{because 'not_move_constructible' does not satisfy '__cpp17_iterator'}} \
// expected-note@#__cpp17_copy_constructible {{because 'not_move_constructible' does not satisfy '__cpp17_copy_constructible'}} \
// expected-note@#__cpp17_move_constructible {{because 'parameter_mapping_regressions::case3::not_move_constructible' does not satisfy '__cpp17_move_constructible'}} \
// expected-note@#is_move_constructible_v {{because 'is_move_constructible_v<parameter_mapping_regressions::case3::not_move_constructible>' evaluated to false}}
}

namespace case4 {

template<bool b>
concept bool_ = b;

template<typename... Ts>
concept unary = bool_<sizeof...(Ts) == 1>;

static_assert(!unary<>);
static_assert(unary<void>);

}

namespace case5 {

template<int size>
concept true1 = size == size;

template<typename... Ts>
concept true2 = true1<sizeof...(Ts)>;

template<typename... Ts>
concept true3 = true2<Ts...>;

static_assert(true3<void>);

}

namespace case6 {

namespace std {
template <int __v>
struct integral_constant {
  static const int value = __v;
};

template <class _Tp, class... _Args>
constexpr bool is_constructible_v = __is_constructible(_Tp, _Args...);

template <class _From, class _To>
constexpr bool is_convertible_v = __is_convertible(_From, _To);

template <class>
struct tuple_size;

template <class _Tp>
constexpr decltype(sizeof(int)) tuple_size_v = tuple_size<_Tp>::value;
}  // namespace std

template <int N, int X>
concept FixedExtentConstructibleFromExtent = X == N;

template <int Extent>
struct span {
  int static constexpr extent = Extent;
  template <typename R, int N = std::tuple_size_v<R>>
    requires(FixedExtentConstructibleFromExtent<extent, N>)
  span(R);
};

template <class, int>
struct array {};

template <class _Tp, decltype(sizeof(int)) _Size>
struct std::tuple_size<array<_Tp, _Size>> : integral_constant<_Size> {};

static_assert(std::is_convertible_v<array<int, 3>, span<3>>);
static_assert(!std::is_constructible_v<span<4>, array<int, 3>>);

}

}
