// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// PR4607
template <class T> struct X {};

template <> struct X<char>
{
  static char* g();
};

template <class T> struct X2 {};

template <class U>
struct X2<U*> {
  static void f() {
    X<U>::g();
  }
};

void a(char *a, char *b) {X2<char*>::f();}

namespace WonkyAccess {
  template<typename T>
  struct X {
    int m;
  };

  template<typename U>
  class Y;

  template<typename U>
  struct Y<U*> : X<U> { };

  template<>
  struct Y<float*> : X<float> { };

  int f(Y<int*> y, Y<float*> y2) {
    return y.m + y2.m;
  }
}

namespace rdar9169404 {
  template<typename T, T N> struct X { };
  template<bool C> struct X<bool, C> {
    typedef int type;
  };

  X<bool, -1>::type value;
#if __cplusplus >= 201103L
  // expected-error@-2 {{non-type template argument evaluates to -1, which cannot be narrowed to type 'bool'}}
#endif
}

namespace rdar39524996 {
  template <typename T, typename U>
  struct enable_if_not_same
  {
    typedef void type;
  };
  template <typename T>
  struct enable_if_not_same<T, T>;

  template <typename T>
  struct Wrapper {
    // Assertion triggered on trying to set twice the same partial specialization
    // enable_if_not_same<int, int>
    template <class U>
    Wrapper(const Wrapper<U>& other,
            typename enable_if_not_same<U, T>::type* = 0) {}

    explicit Wrapper(int i) {}
  };

  template <class T>
  struct Container {
    // It is important that the struct has implicit copy and move constructors.
    Container() : x() {}

    template <class U>
    Container(const Container<U>& other) : x(static_cast<T>(other.x)) {}

    // Implicit constructors are member-wise, so the field triggers instantiation
    // of T constructors and we instantiate all of them for overloading purposes.
    T x;
  };

  void takesWrapperInContainer(const Container< Wrapper<int> >& c);
  void test() {
    // Type mismatch triggers initialization with conversion which requires
    // implicit constructors to be instantiated.
    Container<int> c;
    takesWrapperInContainer(c);
  }
}

namespace InstantiationDependent {
  template<typename> using ignore = void; // expected-warning 0-1{{extension}}
  template<typename T, typename = void> struct A {
    static const bool specialized = false;
  };
  template<typename T> struct Hide { typedef void type; };
  template<typename T> struct A<T, Hide<ignore<typename T::type> >::type> {
    static const bool specialized = true;
  };

  struct X {};
  struct Y { typedef int type; };
  _Static_assert(!A<X>::specialized, "");
  _Static_assert(A<Y>::specialized, "");
}

namespace IgnorePartialSubstitution {
  template <typename... T> struct tuple {}; // expected-warning 0-1{{extension}}
  template <typename> struct IsTuple {
    enum { value = false };
  };
  template <typename... Us> struct IsTuple<tuple<Us...> > { // expected-warning 0-1{{extension}}
    enum { value = true };
  };

  template <bool...> using ignore = void; // expected-warning 0-2{{extension}}
  template <class... Pred> ignore<Pred::value...> helper(); // expected-warning 0-1{{extension}}

  using S = IsTuple<tuple<int> >; // expected-warning 0-1{{extension}}

  // This used to pick the primary template, because we got confused and
  // thought that template parameter 0 was the current partially-substituted
  // pack (from `helper`) during the deduction for the partial specialization.
  void f() { helper<S>(); }

  _Static_assert(S::value, "");
}

namespace GH60778 {
  template <bool B = false> class ClassTemplate {
  public:
      template <typename T, typename = void> class Nested {};
  };

  template <typename DerivedType> class Base {};

  template <>
  template <typename T>
  class ClassTemplate<>::Nested<T> : public Base<ClassTemplate<>::Nested<T> > {};

  void use() {
    // This should instantiate the body of Nested with the template arguments
    // from the Partial Specialization. This would previously get confused and
    // get the template arguments from the primary template instead.
    ClassTemplate<>::Nested<int> instantiation;
  }
}
#if __cplusplus >= 201103L
namespace GH162855 {
  template <class...> using A = int;
  template <class, int> struct B;
  template <class...> struct C;
  template <template <class, int...> class TT, long... X>
  struct C<TT<int, X...>> {
    template <class... Y> using l = A<B<Y, X>...>;
  };
  template <class> struct D;
  template struct C<D<int>>;
} // namespace GH162855
#endif
