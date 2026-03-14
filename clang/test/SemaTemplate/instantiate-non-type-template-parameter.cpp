// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR5311
template<typename T>
class StringSwitch {
public:
  template<unsigned N>
  void Case(const char (&S)[N], const int & Value) {
  }
};

void test_stringswitch(int argc, char *argv[]) {
  (void)StringSwitch<int>();
}

namespace PR6986 {
  template<class Class,typename Type,Type Class::*> 
  struct non_const_member_base
  {
  };

  template<class Class,typename Type,Type Class::*PtrToMember> 
  struct member: non_const_member_base<Class,Type,PtrToMember>
  {
  };

  struct test_class
  {
    int int_member;
  };
  typedef member< test_class,const int,&test_class::int_member > ckey_m;
  void test()
  {
    ckey_m m;
  }
}

namespace rdar8980215 {
  enum E { E1, E2, E3 };

  template<typename T, E e = E2>
  struct X0 { 
    X0() {}
    template<typename U> X0(const X0<U, e> &);
  };

  template<typename T>
  struct X1 : X0<T> { 
    X1() {}
    template<typename U> X1(const X1<U> &x) : X0<T>(x) { }
  };

  X1<int> x1i;
  X1<float> x1f(x1i);
}
