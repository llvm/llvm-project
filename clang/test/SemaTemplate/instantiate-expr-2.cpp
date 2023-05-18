// RUN: %clang_cc1 -fsyntax-only %s
typedef char one_byte;
typedef char (&two_bytes)[2];
typedef char (&four_bytes)[4];
typedef char (&eight_bytes)[8];

template<int N> struct A { };

namespace N1 {
  struct X { };
}

namespace N2 {
  struct Y { };

  two_bytes operator+(Y, Y);
}

namespace N3 {
  struct Z { };

  eight_bytes operator+(Z, Z);
}

namespace N4 {
  one_byte operator+(N1::X, N2::Y);

  template<typename T, typename U>
  struct BinOpOverload {
    typedef A<sizeof(T() + U())> type;
  };
}

namespace N1 {
  four_bytes operator+(X, X);
}

namespace N3 {
  eight_bytes operator+(Z, Z); // redeclaration
}

void test_bin_op_overload(A<1> *a1, A<2> *a2, A<4> *a4, A<8> *a8) {
  typedef N4::BinOpOverload<N1::X, N2::Y>::type XY;
  XY *xy = a1;
  typedef N4::BinOpOverload<N1::X, N1::X>::type XX;
  XX *xx = a4;
  typedef N4::BinOpOverload<N2::Y, N2::Y>::type YY;
  YY *yy = a2;
  typedef N4::BinOpOverload<N3::Z, N3::Z>::type ZZ;
  ZZ *zz = a8;
}

namespace N3 {
  eight_bytes operator-(::N3::Z);
}

namespace N4 {
  template<typename T>
  struct UnaryOpOverload {
    typedef A<sizeof(-T())> type;
  };
}

void test_unary_op_overload(A<8> *a8) {
  typedef N4::UnaryOpOverload<N3::Z>::type UZ;
  UZ *uz = a8;
}

/*
namespace N5 {
  template<int I>
  struct Lookup {
    enum { val = I, more = val + 1 };
  };

  template<bool B>
  struct Cond {
    enum Junk { is = B ? Lookup<B>::more : Lookup<Lookup<B+1>::more>::val };
  };

  enum { resultT = Cond<true>::is,
         resultF = Cond<false>::is };
}
*/

namespace N6 {
  // non-typedependent
  template<int I>
  struct Lookup {};

  template<bool B, typename T, typename E>
  struct Cond {
    typedef Lookup<B ? sizeof(T) : sizeof(E)> True;
    typedef Lookup<!B ? sizeof(T) : sizeof(E)> False;
  };

  typedef Cond<true, int, char>::True True;
  typedef Cond<true, int, char>::False False;

  // check that we have the right types
  Lookup<1> const &L1(False());
  Lookup<sizeof(int)> const &L2(True());
}


namespace N7 {
  // type dependent
  template<int I>
  struct Lookup {};

  template<bool B, typename T, typename E>
  struct Cond {
    T foo() { return B ? T() : E(); }
    typedef Lookup<sizeof(B ? T() : E())> Type;
  };

  //Cond<true, int*, double> C; // Errors
  //int V(C.foo()); // Errors
  //typedef Cond<true, int*, double>::Type Type; // Errors
  typedef Cond<true, int, double>::Type Type;
}

template<typename T, unsigned long N> struct IntegralConstant { };

template<typename T>
struct X0 {
  void f(T x, IntegralConstant<T, sizeof(x)>);
};

void test_X0(X0<int> x, IntegralConstant<int, sizeof(int)> ic) {
  x.f(5,ic);
}

namespace N8 {
  struct X {
    X operator+(const X&) const;
  };
  
  template<typename T>
  T test_plus(const T* xp, const T& x, const T& y) {
    x.operator+(y);
    return xp->operator+(y);
  }
  
  void test_test_plus(X x) {
    test_plus(&x, x, x);
  }
}

namespace N9 {
  struct A {
    bool operator==(int value);
  };
  
  template<typename T> struct B {
    bool f(A a) {
      return a == 1;
    }
  };
  
  template struct B<int>;  
}

namespace N10 {
  template <typename T>
  class A {
    struct X { };
    
  public:
    ~A() {
      f(reinterpret_cast<X *>(0), reinterpret_cast<X *>(0));
    }
    
  private:
    void f(X *);
    void f(X *, X *);
  };
  
  template class A<int>;
}

namespace N12 {
  // PR5224
  template<typename T>
  struct A { typedef int t0; };
  
  struct C  {
    C(int);
    
    template<typename T>
    static C *f0(T a0) {return new C((typename A<T>::t0) 1);   }
  };

  void f0(int **a) { C::f0(a); }
}

namespace PR7202 {
  template<typename U, typename T>
  struct meta {
    typedef T type;
  };

  struct X {
    struct dummy;

    template<typename T>
    X(T, typename meta<T, dummy*>::type = 0);

    template<typename T, typename A>
    X(T, A);
  };

  template<typename T>
  struct Z { };

  template<typename T> Z<T> g(T);

  struct Y {
    template<typename T>
    void f(T t) {
      new X(g(*this));
    }
  };

  template void Y::f(int);
}

namespace N13 {
  class A{
    A(const A&);

  public:
    ~A();
    A(int);
    template<typename T> A &operator<<(const T&);
  };

  template<typename T>
  void f(T t) {
    A(17) << t;
  }

  template void f(int);

}
