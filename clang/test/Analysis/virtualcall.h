#ifdef AS_SYSTEM
#pragma clang system_header

namespace system {
  class A {
  public:
    A() {
      foo(); // no-warning
    }

    virtual int foo();
  };
}

#else

namespace header {
  class A {
  public:
    A() {
      foo();
#if !PUREONLY
#if INTERPROCEDURAL
          // expected-warning-re@-3 {{{{^}}Call Path : fooCall to virtual function during construction will not dispatch to derived class}}
#else
          // expected-warning-re@-5 {{{{^}}Call to virtual function during construction will not dispatch to derived class}}
#endif
#endif

    }

    virtual int foo();
  };
}

#endif
