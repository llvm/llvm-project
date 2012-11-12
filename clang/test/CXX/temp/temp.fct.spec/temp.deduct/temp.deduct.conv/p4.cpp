// RUN: %clang_cc1 -fsyntax-only %s

struct AnyT {
  template<typename T>
  operator T();
};

void test_cvqual_ref(AnyT any) {
  const int &cir = any;  
}

struct AnyThreeLevelPtr {
  template<typename T>
  operator T***() const
  {
    T x = 0;
    // FIXME: looks like we get this wrong, too!
    // x = 0; // will fail if T is deduced to a const type
           // (EDG and GCC get this wrong)
    return 0;
  }
};

struct X { };

void test_deduce_with_qual(AnyThreeLevelPtr a3) {
  int * const * const * const ip = a3;
}

struct AnyPtrMem {
  template<typename Class, typename T>
  operator T Class::*() const
  {
    T x = 0;
    // FIXME: looks like we get this wrong, too!
    // x = 0; // will fail if T is deduced to a const type.
           // (EDG and GCC get this wrong)
    return 0;
  }
};

void test_deduce_ptrmem_with_qual(AnyPtrMem apm) {
  const float X::* pm = apm;
}
