// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics
namespace std_example {
  int i; 
  int f1(); 
  int&& f2(); 
  int &g(const int &);
  float &g(const int &&);
  int &j = g(i); 
  float &k = g(f1());
  float &l = g(f2());

  int &g2(const int &);
  float &g2(int &&);
  int &j2 = g2(i); 
  float &k2 = g2(f1());
  float &l2 = g2(f2());

  // FIXME: We don't support ref-qualifiers yet.
#if 0
  struct A { 
    A& operator<<(int); 
    void p() &; 
    void p() &&;
  };

  A& operator<<(A&&, char); 
  A() << 1; 
  A() << 'c'; 
  A a; 
  a << 1; 
  a << 'c'; 
  A().p(); 
  a.p();
#endif
}

template<typename T>
struct remove_reference {
  typedef T type;
};

template<typename T>
struct remove_reference<T&> {
  typedef T type;
};

template<typename T>
struct remove_reference<T&&> {
  typedef T type;
};

namespace FunctionReferencesOverloading {
  template<typename T> int &f(typename remove_reference<T>::type&);
  template<typename T> float &f(typename remove_reference<T>::type&&);

  void test_f(int (&func_ref)(int)) {
    int &ir = f<int (&)(int)>(func_ref);
  }
}
