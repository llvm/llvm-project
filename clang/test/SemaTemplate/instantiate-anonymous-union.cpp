// RUN: %clang_cc1 -fsyntax-only %s -Wall

template <typename T> class A { struct { }; };

A<int> a0;

template <typename T> struct B {
  union {
    int a;
    void* b;
  };
    
  void f() {
    a = 10;
    b = 0;
  }
};

B<int> b0;

template <typename T> struct C {
  union {
    int a;
    void* b;
  };

  C(int a) : a(a) { }
  C(void* b) : b(b) { }
};

C<int> c0(0);

namespace PR7088 {
  template<typename T>
  void f() { 
    union { 
      int a; 
      union {
        float real;
        T d;
      };
    }; 

    a = 17;
    d = 3.14;
  }

  template void f<double>();
}

// Check for problems related to PR7402 that occur when template instantiation
// instantiates implicit initializers.
namespace PR7402 {
  struct X {
    union {
      struct {
        int x;
        int y;
      };
      int v[2];
    };

    // Check that this requirement survives instantiation.
    template <typename T> X(const T& t) : x(t), y(t) {}
  };

  X x(42.0);
}

namespace PR9188 {
  struct X0 {
    union {
      int member;
    };
  };

  static union {
    int global;
  };

  struct X1 : X0 {
    template<typename T>
    int f() {
      return this->X0::member + PR9188::global;
    }
  };

  template int X1::f<int>();
}
