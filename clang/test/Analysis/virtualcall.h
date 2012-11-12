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
      foo(); // expected-warning{{Call virtual functions during construction or destruction will never go to a more derived class}}
    }

    virtual int foo();
  };
}

#endif
