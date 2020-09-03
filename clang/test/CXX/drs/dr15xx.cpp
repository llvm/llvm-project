// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1512 { // dr1512: 4
  void f(char *p) {
    if (p > 0) {} // expected-error {{ordered comparison between pointer and zero}}
#if __cplusplus >= 201103L
    if (p > nullptr) {} // expected-error {{invalid operands}}
#endif
  }
  bool g(int **x, const int **y) {
    return x < y;
  }

  template<typename T> T val();

  template<typename A, typename B, typename C> void composite_pointer_type_is_base() {
    typedef __typeof(true ? val<A>() : val<B>()) type;
    typedef C type;

    typedef __typeof(val<A>() == val<B>()) cmp;
    typedef __typeof(val<A>() != val<B>()) cmp;
    typedef bool cmp;
  }

  template<typename A, typename B, typename C> void composite_pointer_type_is_ord() {
    composite_pointer_type_is_base<A, B, C>();

    typedef __typeof(val<A>() < val<B>()) cmp;
    typedef __typeof(val<A>() <= val<B>()) cmp;
    typedef __typeof(val<A>() > val<B>()) cmp;
    typedef __typeof(val<A>() >= val<B>()) cmp;
    typedef bool cmp;
  }

  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(int = 0) {
    composite_pointer_type_is_base<A, B, C>();
  }
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() < val<B>()) * = 0);
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() <= val<B>()) * = 0);
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() > val<B>()) * = 0);
  template <typename A, typename B, typename C>
  void composite_pointer_type_is_unord(__typeof(val<A>() >= val<B>()) * = 0);

  // A call to this is ambiguous if a composite pointer type exists.
  template<typename A, typename B>
  void no_composite_pointer_type(__typeof((true ? val<A>() : val<B>()), void()) * = 0);
  template<typename A, typename B> void no_composite_pointer_type(int = 0);

  struct A {};
  struct B : A {};
  struct C {};

  void test() {
#if __cplusplus >= 201103L
    using nullptr_t = decltype(nullptr);
    composite_pointer_type_is_unord<nullptr_t, nullptr_t, nullptr_t>();
    no_composite_pointer_type<nullptr_t, int>();

    composite_pointer_type_is_unord<nullptr_t, const char**, const char**>();
    composite_pointer_type_is_unord<const char**, nullptr_t, const char**>();
#endif

    composite_pointer_type_is_ord<const int *, volatile void *, const volatile void*>();
    composite_pointer_type_is_ord<const void *, volatile int *, const volatile void*>();

    composite_pointer_type_is_ord<const A*, volatile B*, const volatile A*>();
    composite_pointer_type_is_ord<const B*, volatile A*, const volatile A*>();

    composite_pointer_type_is_unord<const int *A::*, volatile int *B::*, const volatile int *const B::*>();
    composite_pointer_type_is_unord<const int *B::*, volatile int *A::*, const volatile int *const B::*>();
    no_composite_pointer_type<int (A::*)(), int (C::*)()>();
    no_composite_pointer_type<const int (A::*)(), volatile int (C::*)()>();

#if __cplusplus > 201402
    composite_pointer_type_is_ord<int (*)() noexcept, int (*)(), int (*)()>();
    composite_pointer_type_is_ord<int (*)(), int (*)() noexcept, int (*)()>();
    composite_pointer_type_is_unord<int (A::*)() noexcept, int (A::*)(), int (A::*)()>();
    composite_pointer_type_is_unord<int (A::*)(), int (A::*)() noexcept, int (A::*)()>();
    // FIXME: This looks like a standard defect; these should probably all have type 'int (B::*)()'.
    composite_pointer_type_is_unord<int (B::*)(), int (A::*)() noexcept, int (B::*)()>();
    composite_pointer_type_is_unord<int (A::*)() noexcept, int (B::*)(), int (B::*)()>();
    composite_pointer_type_is_unord<int (B::*)() noexcept, int (A::*)(), int (B::*)()>();
    composite_pointer_type_is_unord<int (A::*)(), int (B::*)() noexcept, int (B::*)()>();

    // FIXME: It would be reasonable to permit these, with a common type of 'int (*const *)()'.
    no_composite_pointer_type<int (**)() noexcept, int (**)()>();
    no_composite_pointer_type<int (**)(), int (**)() noexcept>();

    // FIXME: It would be reasonable to permit these, with a common type of 'int (A::*)()'.
    no_composite_pointer_type<int (A::*)() const, int (A::*)()>();
    no_composite_pointer_type<int (A::*)(), int (A::*)() const>();

    // FIXME: It would be reasonable to permit these, with a common type of
    // 'int (A::*)() &' and 'int (A::*)() &&', respectively.
    no_composite_pointer_type<int (A::*)() &, int (A::*)()>();
    no_composite_pointer_type<int (A::*)(), int (A::*)() &>();
    no_composite_pointer_type<int (A::*)() &&, int (A::*)()>();
    no_composite_pointer_type<int (A::*)(), int (A::*)() &&>();

    no_composite_pointer_type<int (A::*)() &&, int (A::*)() &>();
    no_composite_pointer_type<int (A::*)() &, int (A::*)() &&>();

    no_composite_pointer_type<int (C::*)(), int (A::*)() noexcept>();
    no_composite_pointer_type<int (A::*)() noexcept, int (C::*)()>();
#endif
  }

#if __cplusplus >= 201103L
  template<typename T> struct Wrap { operator T(); }; // expected-note 4{{converted to type 'nullptr_t'}} expected-note 4{{converted to type 'int *'}}
  void test_overload() {
    using nullptr_t = decltype(nullptr);
    void(Wrap<nullptr_t>() == Wrap<nullptr_t>());
    void(Wrap<nullptr_t>() != Wrap<nullptr_t>());
    void(Wrap<nullptr_t>() < Wrap<nullptr_t>()); // expected-error {{invalid operands}}
    void(Wrap<nullptr_t>() > Wrap<nullptr_t>()); // expected-error {{invalid operands}}
    void(Wrap<nullptr_t>() <= Wrap<nullptr_t>()); // expected-error {{invalid operands}}
    void(Wrap<nullptr_t>() >= Wrap<nullptr_t>()); // expected-error {{invalid operands}}

    // Under dr1213, this is ill-formed: we select the builtin operator<(int*, int*)
    // but then only convert as far as 'nullptr_t', which we then can't convert to 'int*'.
    void(Wrap<nullptr_t>() == Wrap<int*>());
    void(Wrap<nullptr_t>() != Wrap<int*>());
    void(Wrap<nullptr_t>() < Wrap<int*>()); // expected-error {{invalid operands to binary expression ('Wrap<nullptr_t>' and 'Wrap<int *>')}}
    void(Wrap<nullptr_t>() > Wrap<int*>()); // expected-error {{invalid operands}}
    void(Wrap<nullptr_t>() <= Wrap<int*>()); // expected-error {{invalid operands}}
    void(Wrap<nullptr_t>() >= Wrap<int*>()); // expected-error {{invalid operands}}
  }
#endif
}

namespace dr1514 { // dr1514: 11
#if __cplusplus >= 201103L
  struct S {
    enum E : int {}; // expected-note {{previous}}
    enum E : int {}; // expected-error {{redefinition}}
  };
  S::E se; // OK, complete type, not zero-width bitfield.

  // The behavior in other contexts is superseded by DR1966.
#endif
}

namespace dr1518 { // dr1518: 4
#if __cplusplus >= 201103L
struct Z0 { // expected-note 0+ {{candidate}}
  explicit Z0() = default; // expected-note 0+ {{here}}
};
struct Z { // expected-note 0+ {{candidate}}
  explicit Z(); // expected-note 0+ {{here}}
  explicit Z(int); // expected-note {{not a candidate}}
  explicit Z(int, int); // expected-note 0+ {{here}}
};
template <class T> int Eat(T); // expected-note 0+ {{candidate}}
Z0 a;
Z0 b{};
Z0 c = {}; // expected-error {{explicit in copy-initialization}}
int i = Eat<Z0>({}); // expected-error {{no matching function for call to 'Eat'}}

Z c2 = {}; // expected-error {{explicit in copy-initialization}}
int i2 = Eat<Z>({}); // expected-error {{no matching function for call to 'Eat'}}
Z a1 = 1; // expected-error {{no viable conversion}}
Z a3 = Z(1);
Z a2(1);
Z *p = new Z(1);
Z a4 = (Z)1;
Z a5 = static_cast<Z>(1);
Z a6 = {4, 3}; // expected-error {{explicit in copy-initialization}}

struct UserProvidedBaseCtor { // expected-note 0+ {{candidate}}
  UserProvidedBaseCtor() {}
};
struct DoesntInheritCtor : UserProvidedBaseCtor { // expected-note 0+ {{candidate}}
  int x;
};
DoesntInheritCtor I{{}, 42};
#if __cplusplus <= 201402L
// expected-error@-2 {{no matching constructor}}
#endif

struct BaseCtor { BaseCtor() = default; }; // expected-note 0+ {{candidate}}
struct InheritsCtor : BaseCtor { // expected-note 1+ {{candidate}}
  using BaseCtor::BaseCtor;      // expected-note 2 {{inherited here}}
  int x;
};
InheritsCtor II = {{}, 42}; // expected-error {{no matching constructor}}

namespace std_example {
  struct A {
    explicit A() = default; // expected-note 2{{declared here}}
  };

  struct B : A {
    explicit B() = default; // expected-note 2{{declared here}}
  };

  struct C {
    explicit C(); // expected-note 2{{declared here}}
  };

  struct D : A {
    C c;
    explicit D() = default; // expected-note 2{{declared here}}
  };

  template <typename T> void f() {
    T t; // ok
    T u{}; // ok
    T v = {}; // expected-error 4{{explicit}}
  }
  template <typename T> void g() {
    void x(T t); // expected-note 4{{parameter}}
    x({}); // expected-error 4{{explicit}}
  }

  void test() {
    f<A>(); // expected-note {{instantiation of}}
    f<B>(); // expected-note {{instantiation of}}
    f<C>(); // expected-note {{instantiation of}}
    f<D>(); // expected-note {{instantiation of}}
    g<A>(); // expected-note {{instantiation of}}
    g<B>(); // expected-note {{instantiation of}}
    g<C>(); // expected-note {{instantiation of}}
    g<D>(); // expected-note {{instantiation of}}
  }
}
#endif                      // __cplusplus >= 201103L
}

namespace dr1550 { // dr1550: yes
  int f(bool b, int n) {
    return (b ? (throw 0) : n) + (b ? n : (throw 0));
  }
}

namespace dr1560 { // dr1560: 3.5
  void f(bool b, int n) {
    (b ? throw 0 : n) = (b ? n : throw 0) = 0;
  }
  class X { X(const X&); };
  const X &get();
  const X &x = true ? get() : throw 0;
}

namespace dr1563 { // dr1563: yes
#if __cplusplus >= 201103L
  double bar(double) { return 0.0; }
  float bar(float) { return 0.0f; }

  using fun = double(double);
  fun &foo{bar}; // ok
#endif
}

namespace dr1573 { // dr1573: 3.9
#if __cplusplus >= 201103L
  // ellipsis is inherited (p0136r1 supersedes this part).
  struct A { A(); A(int, char, ...); };
  struct B : A { using A::A; };
  B b(1, 'x', 4.0, "hello"); // ok

  // inherited constructor is effectively constexpr if the user-written constructor would be
  struct C { C(); constexpr C(int) {} };
  struct D : C { using C::C; };
  constexpr D d = D(0); // ok
  struct E : C { using C::C; A a; }; // expected-note {{non-literal type}}
  constexpr E e = E(0); // expected-error {{non-literal type}}
  // FIXME: This diagnostic is pretty bad; we should explain that the problem
  // is that F::c would be initialized by a non-constexpr constructor.
  struct F : C { using C::C; C c; }; // expected-note {{here}}
  constexpr F f = F(0); // expected-error {{constant expression}} expected-note {{constructor inherited from base class 'C'}}

  // inherited constructor is effectively deleted if the user-written constructor would be
  struct G { G(int); };
  struct H : G { using G::G; G g; }; // expected-note {{constructor inherited by 'H' is implicitly deleted because field 'g' has no default constructor}}
  H h(0); // expected-error {{constructor inherited by 'H' from base class 'G' is implicitly deleted}}
#endif
}

#if __cplusplus >= 201103L
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
    : __begin_(__b), __size_(__s) {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };

  template < class _T1, class _T2 > struct pair { _T2 second; };

  template<typename T> struct basic_string {
    basic_string(const T* x) {}
    ~basic_string() {};
  };
  typedef basic_string<char> string;

} // std

namespace dr1579 { // dr1579: 3.9
template<class T>
struct GenericMoveOnly {
  GenericMoveOnly();
  template<class U> GenericMoveOnly(const GenericMoveOnly<U> &) = delete; // expected-note 5 {{marked deleted here}}
  GenericMoveOnly(const int &) = delete; // expected-note 2 {{marked deleted here}}
  template<class U> GenericMoveOnly(GenericMoveOnly<U> &&);
  GenericMoveOnly(int &&);
};

GenericMoveOnly<float> DR1579_Eligible(GenericMoveOnly<char> CharMO) {
  int i;
  GenericMoveOnly<char> GMO;

  if (0)
    return i;
  else if (0)
    return GMO;
  else if (0)
    return ((GMO));
  else
    return CharMO;
}

GenericMoveOnly<char> GlobalMO;

GenericMoveOnly<float> DR1579_Ineligible(int &AnInt,
                                          GenericMoveOnly<char> &CharMO) {
  static GenericMoveOnly<char> StaticMove;
  extern GenericMoveOnly<char> ExternMove;

  if (0)
    return AnInt; // expected-error{{invokes a deleted function}}
  else if (0)
    return GlobalMO; // expected-error{{invokes a deleted function}}
  else if (0)
    return StaticMove; // expected-error{{invokes a deleted function}}
  else if (0)
    return ExternMove; // expected-error{{invokes a deleted function}}
  else if (0)
    return AnInt; // expected-error{{invokes a deleted function}}
  else
    return CharMO; // expected-error{{invokes a deleted function}}
}

auto DR1579_lambda_valid = [](GenericMoveOnly<float> mo) ->
  GenericMoveOnly<char> {
  return mo;
};

auto DR1579_lambda_invalid = []() -> GenericMoveOnly<char> {
  static GenericMoveOnly<float> mo;
  return mo; // expected-error{{invokes a deleted function}}
};
} // end namespace dr1579

namespace dr1584 {
  // Deducing function types from cv-qualified types
  template<typename T> void f(const T *); // expected-note {{candidate template ignored}}
  template<typename T> void g(T *, const T * = 0);
  template<typename T> void h(T *) { T::error; } // expected-error {{no members}}
  template<typename T> void h(const T *);
  void i() {
    f(&i); // expected-error {{no matching function}}
    g(&i);
    h(&i); // expected-note {{here}}
  }
}

namespace dr1589 {   // dr1589: 3.7 c++11
  // Ambiguous ranking of list-initialization sequences

  void f0(long, int=0);                 // Would makes selection of #0 ambiguous
  void f0(long);                        // #0
  void f0(std::initializer_list<int>);  // #00
  void g0() { f0({1L}); }               // chooses #00

  void f1(int, int=0);                    // Would make selection of #1 ambiguous
  void f1(int);                           // #1
  void f1(std::initializer_list<long>);   // #2
  void g1() { f1({42}); }                 // chooses #2

  void f2(std::pair<const char*, const char*>, int = 0); // Would makes selection of #3 ambiguous
  void f2(std::pair<const char*, const char*>); // #3
  void f2(std::initializer_list<std::string>);  // #4
  void g2() { f2({"foo","bar"}); }              // chooses #4

  namespace with_error {
    void f0(long);                        // #0
    void f0(std::initializer_list<int>);  // #00     expected-note {{candidate function}}
    void f0(std::initializer_list<int>, int = 0); // expected-note {{candidate function}}
    void g0() { f0({1L}); }                 // expected-error{{call to 'f0' is ambiguous}}

    void f1(int);                           // #1
    void f1(std::initializer_list<long>);   // #2     expected-note {{candidate function}}
    void f1(std::initializer_list<long>, int = 0); // expected-note {{candidate function}}
    void g1() { f1({42}); }                 // expected-error{{call to 'f1' is ambiguous}}

    void f2(std::pair<const char*, const char*>); // #3
    void f2(std::initializer_list<std::string>);  // #4      expected-note {{candidate function}}
    void f2(std::initializer_list<std::string>, int = 0); // expected-note {{candidate function}}
    void g2() { f2({"foo","bar"}); }        // expected-error{{call to 'f2' is ambiguous}}
  }

} // dr1589

namespace dr1591 {  //dr1591. Deducing array bound and element type from initializer list 
  template<class T, int N> int h(T const(&)[N]);
  int X = h({1,2,3});              // T deduced to int, N deduced to 3
  
  template<class T> int j(T const(&)[3]);
  int Y = j({42});                 // T deduced to int, array bound not considered

  struct Aggr { int i; int j; };
  template<int N> int k(Aggr const(&)[N]); //expected-note{{not viable}}
  int Y0 = k({1,2,3});              //expected-error{{no matching function}}
  int Z = k({{1},{2},{3}});        // OK, N deduced to 3

  template<int M, int N> int m(int const(&)[M][N]);
  int X0 = m({{1,2},{3,4}});        // M and N both deduced to 2

  template<class T, int N> int n(T const(&)[N], T);
  int X1 = n({{1},{2},{3}},Aggr()); // OK, T is Aggr, N is 3
  
  
  namespace check_multi_dim_arrays {
    template<class T, int N, int M, int O> int ***f(const T (&a)[N][M][O]); //expected-note{{deduced conflicting values}}
    template<class T, int N, int M> int **f(const T (&a)[N][M]); //expected-note{{couldn't infer}}
   
   template<class T, int N> int *f(const T (&a)[N]); //expected-note{{couldn't infer}}
    int ***p3 = f({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12}  } });
    int ***p33 = f({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12, 13}  } }); //expected-error{{no matching}}
    int **p2 = f({  {1,2,3}, {3, 4, 5}  });
    int **p22 = f({  {1,2}, {3, 4}  });
    int *p1 = f({1, 2, 3});
  }
  namespace check_multi_dim_arrays_rref {
    template<class T, int N, int M, int O> int ***f(T (&&a)[N][M][O]); //expected-note{{deduced conflicting values}}
    template<class T, int N, int M> int **f(T (&&a)[N][M]); //expected-note{{couldn't infer}}
   
    template<class T, int N> int *f(T (&&a)[N]); //expected-note{{couldn't infer}}
    int ***p3 = f({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12}  } });
    int ***p33 = f({  {  {1,2}, {3, 4}  }, {  {5,6}, {7, 8}  }, {  {9,10}, {11, 12, 13}  } }); //expected-error{{no matching}}
    int **p2 = f({  {1,2,3}, {3, 4, 5}  });
    int **p22 = f({  {1,2}, {3, 4}  });
    int *p1 = f({1, 2, 3});
  }
  
  namespace check_arrays_of_init_list {
    template<class T, int N> float *f(const std::initializer_list<T> (&)[N]);
    template<class T, int N> double *f(const T(&)[N]);
    double *p = f({1, 2, 3});
    float *fp = f({{1}, {1, 2}, {1, 2, 3}});
  }
  namespace core_reflector_28543 {
    
    template<class T, int N> int *f(T (&&)[N]);  // #1
    template<class T> char *f(std::initializer_list<T> &&);  //#2
    template<class T, int N, int M> int **f(T (&&)[N][M]); //#3 expected-note{{candidate}}
    template<class T, int N> char **f(std::initializer_list<T> (&&)[N]); //#4 expected-note{{candidate}}

    template<class T> short *f(T (&&)[2]);  //#5

    template<class T> using Arr = T[];
     
    char *pc = f({1, 2, 3}); // OK prefer #2 via 13.3.3.2 [over.ics.rank]
    char *pc2 = f({1, 2}); // #2 also 
    int *pi = f(Arr<int>{1, 2, 3}); // OK prefer #1

    void *pv1 = f({ {1, 2, 3}, {4, 5, 6} }); // expected-error{{ambiguous}} btw 3 & 4
    char **pcc = f({ {1}, {2, 3} }); // OK #4

    short *ps = f(Arr<int>{1, 2});  // OK #5
  }
} // dr1591

#endif
