// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR5908
template <typename Iterator>
void Test(Iterator it) {
  *(it += 1);
}

namespace PR6045 {
  template<unsigned int r>
  class A
  {
    static const unsigned int member = r;
    void f();
  };
  
  template<unsigned int r>
  const unsigned int A<r>::member;
  
  template<unsigned int r>
  void A<r>::f() 
  {
    unsigned k;
    (void)(k % member);
  }
}

namespace PR7198 {
  struct A
  {
    ~A() { }
  };

  template<typename T>
  struct B {
    struct C : A {};
    void f()
    {
      C c = C();
    }
  };
}

namespace PR7724 {
  template<typename OT> int myMethod()
  { return 2 && sizeof(OT); }
}

namespace test4 {
  template <typename T> T *addressof(T &v) {
    return reinterpret_cast<T*>(
             &const_cast<char&>(reinterpret_cast<const volatile char &>(v)));
  }
}

namespace test5 {
  template <typename T> class chained_map {
    int k;
    void lookup() const {
      int &v = (int &)k;
    }
  };
}

namespace PR8795 {
  template <class _CharT> int test(_CharT t)
  {
    int data [] = {
      sizeof(_CharT) > sizeof(char)
    };
    return data[0];
  }
}
