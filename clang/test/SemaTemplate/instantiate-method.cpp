// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
class X {
public:
  void f(T x); // expected-error{{argument may not have 'void' type}}
  void g(T*);

  static int h(T, T); // expected-error {{argument may not have 'void' type}}
};

int identity(int x) { return x; }

void test(X<int> *xi, int *ip, X<int(int)> *xf) {
  xi->f(17);
  xi->g(ip);
  xf->f(&identity);
  xf->g(identity);
  X<int>::h(17, 25);
  X<int(int)>::h(identity, &identity);
}

void test_bad() {
  X<void> xv; // expected-note{{in instantiation of template class 'X<void>' requested here}}
}

template<typename T, typename U>
class Overloading {
public:
  int& f(T, T); // expected-note{{previous declaration is here}}
  float& f(T, U); // expected-error{{functions that differ only in their return type cannot be overloaded}}
};

void test_ovl(Overloading<int, long> *oil, int i, long l) {
  int &ir = oil->f(i, i);
  float &fr = oil->f(i, l);
}

void test_ovl_bad() {
  Overloading<float, float> off; // expected-note{{in instantiation of template class 'Overloading<float, float>' requested here}}
}

template<typename T>
class HasDestructor {
public:
  virtual ~HasDestructor() = 0;
};

int i = sizeof(HasDestructor<int>); // FIXME: forces instantiation, but 
                // the code below should probably instantiate by itself.
int abstract_destructor[__is_abstract(HasDestructor<int>)? 1 : -1];


template<typename T>
class Constructors {
public:
  Constructors(const T&);
  Constructors(const Constructors &other);
};

void test_constructors() {
  Constructors<int> ci1(17);
  Constructors<int> ci2 = ci1;
}


template<typename T>
struct ConvertsTo {
  operator T();
};

void test_converts_to(ConvertsTo<int> ci, ConvertsTo<int *> cip) {
  int i = ci;
  int *ip = cip;
}

// PR4660
template<class T> struct A0 { operator T*(); };
template<class T> struct A1;

int *a(A0<int> &x0, A1<int> &x1) {
  int *y0 = x0;
  int *y1 = x1; // expected-error{{no viable conversion}}
}

struct X0Base {
  int &f();
  int& g(int);
  static double &g(double);
};

template<typename T>
struct X0 : X0Base {
};

template<typename U>
struct X1 : X0<U> {
  int &f2() { 
    return X0Base::f();
  }
};

void test_X1(X1<int> x1i) {
  int &ir = x1i.f2();
}

template<typename U>
struct X2 : X0Base, U {
  int &f2() { return X0Base::f(); }
};

template<typename T>
struct X3 {
  void test(T x) {
    double& d1 = X0Base::g(x);
  }
};


template struct X3<double>;

// Don't try to instantiate this, it's invalid.
namespace test1 {
  template <class T> class A {};
  template <class T> class B {
    void foo(A<test1::Undeclared> &a) // expected-error {{no member named 'Undeclared' in namespace 'test1'}}
    {}
  };
  template class B<int>;
}

namespace PR6947 {
  template< class T > 
  struct X {
    int f0( )      
    {
      typedef void ( X::*impl_fun_ptr )( );
      impl_fun_ptr pImpl = &X::template
        f0_impl1<int>;
    }
  private:                  
    int f1() {
    }
    template< class Processor>                  
    void f0_impl1( )                 
    {
    }
  };

  char g0() {
    X<int> pc;
    pc.f0();
  }

}

namespace PR7022 {
  template <typename > 
  struct X1
  {
    typedef int state_t( );
    state_t g ;
  };

  template <  typename U = X1<int> > struct X2
  {
    X2( U = U())
    {
    }
  };

  void m(void)
  {
    typedef X2<> X2_type;
    X2_type c;
  }
}

namespace SameSignatureAfterInstantiation {
  template<typename T> struct S {
    void f(T *); // expected-note {{previous}}
    void f(const T*); // expected-error-re {{multiple overloads of 'f' instantiate to the same signature 'void (const int *){{( __attribute__\(\(thiscall\)\))?}}'}}
  };
  S<const int> s; // expected-note {{instantiation}}
}

namespace PR22040 {
  template <typename T> struct Foobar {
    template <> void bazqux(typename T::type) {}  // expected-error {{cannot specialize a function 'bazqux' within class scope}} expected-error 2{{cannot be used prior to '::' because it has no members}}
  };

  void test() {
    // FIXME: we should suppress the "no member" errors
    Foobar<void>::bazqux();  // expected-error{{no member named 'bazqux' in }}  expected-note{{in instantiation of template class }}
    Foobar<int>::bazqux();  // expected-error{{no member named 'bazqux' in }}  expected-note{{in instantiation of template class }}
    Foobar<int>::bazqux(3);  // expected-error{{no member named 'bazqux' in }}
  }
}

template <typename>
struct SpecializationOfGlobalFnInClassScope {
  template <>
  void ::Fn(); // expected-error{{cannot have a qualified name}}
};

class AbstractClassWithGlobalFn {
  template <typename>
  void ::f(); // expected-error{{cannot have a qualified name}}
  virtual void f1() = 0;
};
