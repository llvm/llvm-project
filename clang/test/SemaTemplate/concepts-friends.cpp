// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template <typename T>
concept constraint = false;
namespace temp_friend_9 {
// A non-template friend declaration with a requires-clause shall be a
// definition. ...Such a constrained friend function ... does not declare the
// same function or function template as a declaration in any other scope.
template <typename T>
struct NonTemplateFriend {
  friend void foo()
    requires true
  {}
};

// A friend function template with a constraint that depends on a template
// parameter from an enclosing template shall be a definition.  Such a ...
// function template declaration does not declare the same function or
// function template as a declaration in any other scope.
template <typename T>
struct TemplateFromEnclosing {
  template <typename U>
  friend void foo()
    requires constraint<T>
  {}

  T variable;
  template <typename U>
  friend void foo2()
    requires constraint<decltype(variable)>
  {}

  template <typename U>
  friend void foo3(T parmvar)
    requires constraint<decltype(parmvar)>
  {}

  template <typename U>
  friend void foo4()
    requires requires(T &req) { (void)req; }
  {}

  using Alias = T;
  template <typename U>
  friend void foo5()
    requires constraint<Alias>
  {}

  // All of these refer to a parent, so these are not duplicate definitions.
  struct ChildOfEnclosing {
    template <typename U>
    friend void foo6()
      requires constraint<T>
    {}
    template <typename U>
    friend void foo7()
      requires constraint<decltype(variable)>
    {}
    template <typename U>
    friend void foo8(T parmvar)
      requires constraint<decltype(parmvar)>
    {}
    // This is NOT a duplicate since it itself is not a template.
    friend void foo9()
      requires true
    {}
  };
  template <typename T2>
  struct TemplChildOfEnclosing {
    template <typename U>
    friend void foo10()
      requires constraint<T>
    {}
  };
};

// Doesn't meet either of the requirements in the above as they don't refer to
// an enclosing scope.
template <typename T>
struct Redefinition {
  template <typename U>
  friend void foo() // #REDEF
    requires constraint<U>
  {}

  struct ChildOfRedef {
    template <typename U>
    friend void foo2() // #REDEF2
      requires constraint<U>
    {}
  };
  template <typename T2>
  struct ChildOfRedef2 {
    template <typename U>
    friend void foo3() // #REDEF3
      requires constraint<U>
    {}
  };
};

void bar() {
  NonTemplateFriend<int> S1;
  NonTemplateFriend<float> S2;
  TemplateFromEnclosing<int> S3;
  TemplateFromEnclosing<int>::ChildOfEnclosing S3b;
  TemplateFromEnclosing<float> S4;
  TemplateFromEnclosing<float>::ChildOfEnclosing S4b;
  Redefinition<int> S5;
  Redefinition<float> S6;
  // expected-error@#REDEF {{redefinition of 'foo'}}
  // expected-note@-2{{in instantiation of template class }}
  // expected-note@#REDEF {{previous definition is here}}
  Redefinition<int>::ChildOfRedef S7;
  Redefinition<float>::ChildOfRedef S8;
  // expected-error@#REDEF2 {{redefinition of 'foo2'}}
  // expected-note@-2{{in instantiation of member class }}
  // expected-note@#REDEF2 {{previous definition is here}}

  Redefinition<int>::ChildOfRedef2<int> S9;
  Redefinition<float>::ChildOfRedef2<float> S10;
  // expected-error@#REDEF3 {{redefinition of 'foo3'}}
  // expected-note@-2{{in instantiation of template class }}
  // expected-note@#REDEF3 {{previous definition is here}}
}
} // namespace temp_friend_9

namespace SameScopeRedefs {
template <typename T>
struct NonTemplateFriend {
  friend void foo() // #NTF1
    requires true
  {}
  friend void foo() // #NTF2
    requires true
  {}
};

template <typename T>
struct TemplateFromEnclosing {
  template <typename U>
  friend void foo() // #TFE1
    requires constraint<T>
  {}
  template <typename U>
  friend void foo() // #TFE2
    requires constraint<T>
  {}
};
// Same as above, but doesn't require an instantiation pair to cause.
template <typename T>
struct Redefinition {
  template <typename U>
  friend void foo() // #RD1
    requires constraint<U>
  {}
  template <typename U>
  friend void foo() // #RD2
    requires constraint<U>
  {}
};
void bar() {
  NonTemplateFriend<int> S1;
  // expected-error@#NTF2 {{redefinition of 'foo'}}
  // expected-note@-2{{in instantiation of template class}}
  // expected-note@#NTF1 {{previous definition is here}}

  TemplateFromEnclosing<int> S2;
  // expected-error@#TFE2 {{redefinition of 'foo'}}
  // expected-note@-2{{in instantiation of template class}}
  // expected-note@#TFE1 {{previous definition is here}}

  Redefinition<int> S3;
  // expected-error@#RD2 {{redefinition of 'foo'}}
  // expected-note@-2{{in instantiation of template class}}
  // expected-note@#RD1 {{previous definition is here}}
}
} // namespace SameScopeRedefs

namespace LibCXXOperatorRedef {
template <typename T, typename U> struct is_same {
  static constexpr bool value = false;
};
template <typename T> struct is_same<T, T> {
  static constexpr bool value = false;
};

template <typename T, typename U>
concept same_as = is_same<T, U>::value;

// An issue found from libcxx when trying to commit the deferred concepts patch.
// This caused an error of 'redefinition of funcN'.
template <class _Tp> struct __range_adaptor_closure {
  template <typename _View, typename _Closure>
    requires same_as<_Tp, _Closure>
  friend constexpr decltype(auto) R1func1(_View &&__view,
                                          _Closure &&__closure){};
  template <typename _View, typename _Closure>
  friend constexpr decltype(auto) R1func2(_View &&__view,
                                          _Closure &&__closure)
    requires same_as<_Tp, _Closure>
  {};
  template <same_as<_Tp> _View, typename _Closure>
  friend constexpr decltype(auto) R1func3(_View &&__view,
                                          _Closure &&__closure){};
};

struct A : __range_adaptor_closure<A> {};
struct B : __range_adaptor_closure<B> {};

// These three fail because after the 1st pass of instantiation, they are still
// identical.
template <class _Tp> struct __range_adaptor_closure2 {
  template <typename _View, typename _Closure>
    requires same_as<_View, _Closure>
  friend constexpr decltype(auto) R2func1(_View &&__view, // #FUNC1
                                          _Closure &&__closure){};
  template <typename _View, typename _Closure>
  friend constexpr decltype(auto) R2func2(_View &&__view, // #FUNC2
                                          _Closure &&__closure)
    requires same_as<_View, _Closure>
  {};
  template <typename _View, same_as<_View> _Closure>
  friend constexpr decltype(auto) R2func3(_View &&__view, // #FUNC3
                                          _Closure &&__closure){};
};

struct A2 : __range_adaptor_closure2<A2> {};
struct B2 : __range_adaptor_closure2<B2> {};
// expected-error@#FUNC1{{redefinition of 'R2func1'}}
// expected-note@-2{{in instantiation of template class}}
// expected-note@#FUNC1{{previous definition is here}}
// expected-error@#FUNC2{{redefinition of 'R2func2'}}
// expected-note@#FUNC2{{previous definition is here}}
// expected-error@#FUNC3{{redefinition of 'R2func3'}}
// expected-note@#FUNC3{{previous definition is here}}

// These three are fine, they all depend on the parent template parameter, so
// are different despite ::type not being valid.
template <class _Tp> struct __range_adaptor_closure3 {
  template <typename _View, typename _Closure>
    requires same_as<typename _Tp::type, _Closure>
  friend constexpr decltype(auto) R3func1(_View &&__view,
                                          _Closure &&__closure){};
  template <typename _View, typename _Closure>
  friend constexpr decltype(auto) R3func2(_View &&__view,
                                          _Closure &&__closure)
    requires same_as<typename _Tp::type, _Closure>
  {};
  template <same_as<typename _Tp::type> _View, typename _Closure>
  friend constexpr decltype(auto) R3func3(_View &&__view,
                                          _Closure &&__closure){};
};

struct A3 : __range_adaptor_closure3<A3> {};
struct B3 : __range_adaptor_closure3<B3> {};

template <class _Tp> struct __range_adaptor_closure4 {
  template <typename _View, typename _Closure>
    requires same_as<_Tp, _View>
  // expected-note@+1{{previous definition is here}}
  void foo1(_View &&, _Closure &&) {}
  template <typename _View, typename _Closure>
    requires same_as<_Tp, _View>
  // expected-error@+1{{class member cannot be redeclared}}
  void foo1(_View &&, _Closure &&) {}

  template <typename _View, typename _Closure>
  // expected-note@+1{{previous definition is here}}
  void foo2(_View &&, _Closure &&)
    requires same_as<_Tp, _View>
  {}
  template <typename _View, typename _Closure>
  // expected-error@+1{{class member cannot be redeclared}}
  void foo2(_View &&, _Closure &&)
    requires same_as<_Tp, _View>
  {}

  template <same_as<_Tp> _View, typename _Closure>
  // expected-note@+1{{previous definition is here}}
  void foo3(_View &&, _Closure &&) {}
  template <same_as<_Tp> _View, typename _Closure>
  // expected-error@+1{{class member cannot be redeclared}}
  void foo3(_View &&, _Closure &&) {}
};

// Requires instantiation to fail, so no errors here.
template <class _Tp> struct __range_adaptor_closure5 {
  template <same_as<_Tp> U>
  friend void foo() {}
  template <same_as<_Tp> U>
  friend void foo() {}
};

template <class _Tp> struct __range_adaptor_closure6 {
  template <same_as<_Tp> U>
  friend void foo() {} // #RAC6FOO1
  template <same_as<_Tp> U>
  friend void foo() {} // #RAC6FOO2
};
struct A6 : __range_adaptor_closure6<A6> {};
// expected-error@#RAC6FOO2{{redefinition of 'foo'}}
// expected-note@-2{{in instantiation of template class}}
// expected-note@#RAC6FOO1{{previous definition is here}}

template <class T> struct S1 {
  template <typename U>
  friend void dupe() {} // #S1DUPE

  template <typename U>
    requires same_as<U, U>
  friend void dupe2() {} // #S1DUPE2
};
template <class T> struct S2 {
  template <typename U>
  friend void dupe() {} // #S2DUPE

  template <typename U>
    requires same_as<U, U>
  friend void dupe2() {} // #S2DUPE2
};

template <class T> struct S3 {
  template <typename U>
    requires same_as<T, U>
  friend void dupe() {}
};
template <class T> struct S4 {
  template <typename U>
    requires same_as<T, U>
  friend void dupe() {}
};

// Same as S3 and S4, but aren't instantiated with the same T.
template <class T> struct S5 {
  template <typename U>
    requires same_as<T, U>
  friend void not_dupe() {}
};
template <class T> struct S6 {
  template <typename U>
    requires same_as<T, U>
  friend void not_dupe() {}
};

template <class T> struct S7 {
  void not_dupe()
    requires same_as<T, T>
  {}
};

void useS() {
  S1<int> s1;
  S2<double> s2;
  // expected-error@#S2DUPE{{redefinition}}
  // expected-note@-2{{in instantiation of template class}}
  // expected-note@#S1DUPE{{previous definition is here}}
  // expected-error@#S2DUPE2{{redefinition}}
  // expected-note@#S1DUPE2{{previous definition is here}}

  // OK, they have different 'scopes'.
  S3<int> s3;
  S4<int> s4;

  // OK, because only instantiated with different T.
  S5<int> s5;
  S6<double> s6;

  S7<int> s7;
}

} // namespace LibCXXOperatorRedef

namespace NamedDeclRefs {
  namespace my_std {
    template<typename T, typename U>
      concept Outer = true;
    template<typename T>
      using Inner = T;
  }
  template<typename T>
    struct Proxy {
      template<class U>
        friend constexpr void RefOuter()
        requires my_std::Outer<my_std::Inner<T>, my_std::Inner<U>>{}
      template<class U>
        friend constexpr void NoRefOuter() // #NOREFOUTER
        requires my_std::Outer<my_std::Inner<U>, my_std::Inner<U>>{}
    };
  void use() {
    Proxy<int> p;
    Proxy<float> p2;
    // expected-error@#NOREFOUTER {{redefinition of 'NoRefOuter'}}
    // expected-note@-2{{in instantiation of template class}}
    // expected-note@#NOREFOUTER{{previous definition is here}}
  }
} // namespace NamedDeclRefs

namespace RefersToParentInConstraint {
  // No diagnostic, these aren't duplicates.
  template<typename T, typename U>
  concept similar = true;

  template <typename X>
  struct S{
    friend void f(similar<S> auto && self){}
    friend void f2(similar<S<X>> auto && self){}
  };

  void use() {
    S<int> x;
    S<long> y;
  }
} // namespace RefersToParentInConstraint
