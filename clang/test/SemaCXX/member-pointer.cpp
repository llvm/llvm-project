// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx98- %s

struct A {};
enum B { Dummy };
namespace C {}
struct D : A {};
struct E : A {};
struct F : D, E {};
struct G : virtual D {};
class H : A {}; // expected-note 2{{implicitly declared private here}}

int A::*pdi1;
int (::A::*pdi2);
int (A::*pfi)(int);
void (*A::*ppfie)() throw();
#if __cplusplus < 201703L
// expected-error@-2 {{exception specifications are not allowed beyond a single level of indirection}}
#endif

int B::*pbi;
#if __cplusplus <= 199711L // C++03 or earlier modes
// expected-warning@-2 {{use of enumeration in a nested name specifier is a C++11 extension}}
#endif
// expected-error@-4 {{'pbi' does not point into a class}}
int C::*pci; // expected-error {{'pci' does not point into a class}}
void A::*pdv; // expected-error {{'pdv' declared as a member pointer to void}}
int& A::*pdr; // expected-error {{'pdr' declared as a member pointer to a reference}}

void f() {
  // This requires tentative parsing.
  int (A::*pf)(int, int);

  // Implicit conversion to bool.
  bool b = pdi1;
  b = pfi;

  // Conversion from null pointer constant.
  pf = 0;
  pf = __null;

  // Conversion to member of derived.
  int D::*pdid = pdi1;
  pdid = pdi2;

  // Fail conversion due to ambiguity and virtuality.
  int F::*pdif = pdi1; // expected-error {{ambiguous conversion from pointer to member of base class 'A' to pointer to member of derived class 'F':}}
  int G::*pdig = pdi1; // expected-error {{conversion from pointer to member of class 'A' to pointer to member of class 'G' via virtual base 'D' is not allowed}}

  // Conversion to member of base.
  pdi1 = pdid; // expected-error {{assigning to 'int A::*' from incompatible type 'int D::*'}}
  
  // Comparisons
  int (A::*pf2)(int, int);
  int (D::*pf3)(int, int) = 0;
  bool b1 = (pf == pf2); (void)b1;
  bool b2 = (pf != pf2); (void)b2;
  bool b3 = (pf == pf3); (void)b3;
  bool b4 = (pf != 0); (void)b4;
}

struct TheBase
{
  void d();
};

struct HasMembers : TheBase
{
  int i;
  void f();

  void g();
  void g(int);
  static void g(double);
};

namespace Fake
{
  int i;
  void f();
}

void g() {
  HasMembers hm;

  int HasMembers::*pmi = &HasMembers::i;
  int *pni = &Fake::i;
  int *pmii = &hm.i;

  void (HasMembers::*pmf)() = &HasMembers::f;
  void (*pnf)() = &Fake::f;
  &hm.f; // expected-error {{cannot create a non-constant pointer to member function}}

  void (HasMembers::*pmgv)() = &HasMembers::g;
  void (HasMembers::*pmgi)(int) = &HasMembers::g;
  void (*pmgd)(double) = &HasMembers::g;

  void (HasMembers::*pmd)() = &HasMembers::d;
}

struct Incomplete;

void h() {
  HasMembers hm, *phm = &hm;

  int HasMembers::*pi = &HasMembers::i;
  hm.*pi = 0;
  int i = phm->*pi;
  (void)&(hm.*pi);
  (void)&(phm->*pi);
  (void)&((&hm)->*pi); 

  void (HasMembers::*pf)() = &HasMembers::f;
  (hm.*pf)();
  (phm->*pf)();

  (void)(hm->*pi); // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'HasMembers'}}
  (void)(phm.*pi); // expected-error {{left hand operand to .* must be a class compatible with the right hand operand, but is 'HasMembers *'}}
  (void)(i.*pi); // expected-error {{left hand operand to .* must be a class compatible with the right hand operand, but is 'int'}}
  int *ptr;
  (void)(ptr->*pi); // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'int *'}}

  int A::*pai = 0;
  D d, *pd = &d;
  (void)(d.*pai);
  (void)(pd->*pai);
  F f, *ptrf = &f;
  (void)(f.*pai); // expected-error {{ambiguous conversion from derived class 'F' to base class 'A'}}
  (void)(ptrf->*pai); // expected-error {{ambiguous conversion from derived class 'F' to base class 'A'}}
  H h, *ptrh = &h;
  (void)(h.*pai); // expected-error {{cannot cast 'H' to its private base class 'A'}}
  (void)(ptrh->*pai); // expected-error {{cannot cast 'H' to its private base class 'A'}}

  (void)(hm.*i); // expected-error {{pointer-to-member}}
  (void)(phm->*i); // expected-error {{pointer-to-member}}

  // Okay
  Incomplete *inc;
  int Incomplete::*pii = 0;
  (void)(inc->*pii);
}

struct OverloadsPtrMem
{
  int operator ->*(const char *);
};

void i() {
  OverloadsPtrMem m;
  int foo = m->*"Awesome!";
}

namespace pr5985 {
  struct c {
    void h();
    void f() {
      void (c::*p)();
      p = &h; // expected-error {{must explicitly qualify}}
      p = &this->h; // expected-error {{cannot create a non-constant pointer to member function}}
      p = &(*this).h; // expected-error {{cannot create a non-constant pointer to member function}}
    }
  };
}

namespace pr6783 {
  struct Base {};
  struct X; // expected-note {{forward declaration}}

  int test1(int Base::* p2m, X* object)
  {
    return object->*p2m; // expected-error {{left hand operand to ->*}}
  }
}

namespace PR7176 {
  namespace base
  {
    struct Process
    { };
    struct Continuous : Process
    {
      bool cond();
    };
  }

  typedef bool( base::Process::*Condition )();

  void m()
  { (void)(Condition) &base::Continuous::cond; }
}

namespace rdar8358512 {
  // We can't call this with an overload set because we're not allowed
  // to look into overload sets unless the parameter has some kind of
  // function type.
  template <class F> void bind(F f); // expected-note 12 {{candidate template ignored}}
  template <class F, class T> void bindmem(F (T::*f)()); // expected-note 4 {{candidate template ignored}}
  template <class F> void bindfn(F (*f)()); // expected-note 4 {{candidate template ignored}}

  struct A {
    void nonstat();
    void nonstat(int);

    void mixed();
    static void mixed(int);

    static void stat();
    static void stat(int);
    
    template <typename T> struct Test0 {
      void test() {
        bind(&nonstat); // expected-error {{no matching function for call}}
        bind(&A::nonstat); // expected-error {{no matching function for call}}

        bind(&mixed); // expected-error {{no matching function for call}}
        bind(&A::mixed); // expected-error {{no matching function for call}}

        bind(&stat); // expected-error {{no matching function for call}}
        bind(&A::stat); // expected-error {{no matching function for call}}
      }
    };

    template <typename T> struct Test1 {
      void test() {
        bindmem(&nonstat); // expected-error {{no matching function for call}}
        bindmem(&A::nonstat);

        bindmem(&mixed); // expected-error {{no matching function for call}}
        bindmem(&A::mixed);

        bindmem(&stat); // expected-error {{no matching function for call}}
        bindmem(&A::stat); // expected-error {{no matching function for call}}
      }
    };

    template <typename T> struct Test2 {
      void test() {
        bindfn(&nonstat); // expected-error {{no matching function for call}}
        bindfn(&A::nonstat); // expected-error {{no matching function for call}}

        bindfn(&mixed); // expected-error {{no matching function for call}}
        bindfn(&A::mixed); // expected-error {{no matching function for call}}

        bindfn(&stat);
        bindfn(&A::stat);
      }
    };
  };

  template <class T> class B {
    void nonstat();
    void nonstat(int);

    void mixed();
    static void mixed(int);

    static void stat();
    static void stat(int);

    // None of these can be diagnosed yet, because the arguments are
    // still dependent.
    void test0a() {
      bind(&nonstat);
      bind(&B::nonstat);

      bind(&mixed);
      bind(&B::mixed);

      bind(&stat);
      bind(&B::stat);
    }

    void test0b() {
      bind(&nonstat); // expected-error {{no matching function for call}}
      bind(&B::nonstat); // expected-error {{no matching function for call}}

      bind(&mixed); // expected-error {{no matching function for call}}
      bind(&B::mixed); // expected-error {{no matching function for call}}

      bind(&stat); // expected-error {{no matching function for call}}
      bind(&B::stat); // expected-error {{no matching function for call}}
    }
  };

  template void B<int>::test0b(); // expected-note {{in instantiation}}
}

namespace PR9973 {
  template<class R, class T> struct dm
  {
    typedef R T::*F;
    F f_;
    template<class U> int & call(U u)
    { return u->*f_; } // expected-error{{reference to non-static member function must be called; did you mean to call it with no arguments?}} expected-error {{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'int'}}

    template<class U> int operator()(U u)
    { call(u); } // expected-note{{in instantiation of}}
  };

  template<class R, class T> 
  dm<R, T> mem_fn(R T::*) ;

  struct test
  { int nullary_v(); };

  void f()
  {
    test* t;
    mem_fn(&test::nullary_v)(t); // expected-note{{in instantiation of}}
  }
}

namespace test8 {
  struct A { int foo; };
  int test1() {
    // Verify that we perform (and check) an lvalue conversion on the operands here.
    return (*((A**) 0)) // expected-warning {{indirection of non-volatile null pointer will be deleted}} expected-note {{consider}}
             ->**(int A::**) 0; // expected-warning {{indirection of non-volatile null pointer will be deleted}} expected-note {{consider}}
  }

  int test2() {
    // Verify that we perform (and check) an lvalue conversion on the operands here.
    // TODO: the .* should itself warn about being a dereference of null.
    return (*((A*) 0))
             .**(int A::**) 0; // expected-warning {{indirection of non-volatile null pointer will be deleted}} expected-note {{consider}}
  }
}
