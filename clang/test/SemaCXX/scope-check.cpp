// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -fcxx-exceptions %s -Wno-unreachable-code
// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -fcxx-exceptions -std=gnu++11 %s -Wno-unreachable-code

namespace test0 {
  struct D { ~D(); };

  int f(bool b) {
    if (b) {
      D d;
      goto end;
    }

  end:
    return 1;
  }
}

namespace test1 {
  struct C { C(); };

  int f(bool b) {
    if (b)
      goto foo; // expected-error {{goto into protected scope}}
    C c; // expected-note {{jump bypasses variable initialization}}
  foo:
    return 1;
  }
}

namespace test2 {
  struct C { C(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };

    C c;
    goto *ip;
  lbl1:
    return 0;
  lbl2:
    return 1;
  }
}

namespace test3 {
  struct C { C(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };

    goto *ip;
  lbl1: {
    C c;
    return 0;
  }
  lbl2:
    return 1;
  }
}

namespace test4 {
  struct C { C(); };
  struct D { ~D(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };

    C c0;

    goto *ip; // expected-error {{indirect goto might cross protected scopes}}
    C c1; // expected-note {{jump bypasses variable initialization}}
  lbl1: // expected-note {{possible target of indirect goto}}
    return 0;
  lbl2:
    return 1;
  }
}

namespace test5 {
  struct C { C(); };
  struct D { ~D(); };

  int f(void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2 };
    C c0;

    goto *ip;
  lbl1: // expected-note {{possible target of indirect goto}}
    return 0;
  lbl2:
    if (ip[1]) {
      D d; // expected-note {{jump exits scope of variable with non-trivial destructor}}
      ip += 2;
      goto *ip; // expected-error {{indirect goto might cross protected scopes}}
    }
    return 1;
  }
}

namespace test6 {
  struct C { C(); };

  unsigned f(unsigned s0, unsigned s1, void **ip) {
    static void *ips[] = { &&lbl1, &&lbl2, &&lbl3, &&lbl4 };
    C c0;

    goto *ip;
  lbl1:
    s0++;
    goto *++ip;
  lbl2:
    s0 -= s1;
    goto *++ip;
  lbl3: {
    unsigned tmp = s0;
    s0 = s1;
    s1 = tmp;
    goto *++ip;
  }
  lbl4:
    return s0;
  }
}

// C++0x says it's okay to skip non-trivial initializers on static
// locals, and we implement that in '03 as well.
namespace test7 {
  struct C { C(); };

  void test() {
    goto foo;
    static C c;
  foo:
    return;
  }
}

// PR7789
namespace test8 {
  void test1(int c) {
    switch (c) {
    case 0:
      int x = 56; // expected-note {{jump bypasses variable initialization}}
    case 1:       // expected-error {{switch case is in protected scope}}
      x = 10;
    }
  }

  void test2() {
    goto l2;     // expected-error {{goto into protected scope}}
  l1: int x = 5; // expected-note {{jump bypasses variable initialization}}
  l2: x++;
  }
}

namespace test9 {
  struct S { int i; };
  void test1() {
    goto foo;
    S s;
  foo:
    return;
  }
  unsigned test2(unsigned x, unsigned y) {
    switch (x) {
    case 2:
      S s;
      if (y > 42) return x + y;
    default:
      return x - 2;
    }
  }
}

// http://llvm.org/PR10462
namespace PR10462 {
  enum MyEnum {
    something_valid,
    something_invalid
  };

  bool recurse() {
    MyEnum K;
    switch (K) { // expected-warning {{enumeration value 'something_invalid' not handled in switch}}
    case something_valid:
    case what_am_i_thinking: // expected-error {{use of undeclared identifier}}
      int *X = 0;
      if (recurse()) {
      }

      break;
    }
  }
}

namespace test10 {
  int test() {
    static void *ps[] = { &&a0 };
    goto *&&a0; // expected-error {{goto into protected scope}}
    int a = 3; // expected-note {{jump bypasses variable initialization}}
  a0:
    return 0;
  }
}

// pr13812
namespace test11 {
  struct C {
    C(int x);
    ~C();
  };
  void f(void **ip) {
    static void *ips[] = { &&l0 };
  l0:  // expected-note {{possible target of indirect goto}}
    C c0 = 42; // expected-note {{jump exits scope of variable with non-trivial destructor}}
    goto *ip; // expected-error {{indirect goto might cross protected scopes}}
  }
}

namespace test12 {
  struct C {
    C(int x);
    ~C();
  };
  void f(void **ip) {
    static void *ips[] = { &&l0 };
    const C c0 = 17;
  l0: // expected-note {{possible target of indirect goto}}
    const C &c1 = 42; // expected-note {{jump exits scope of variable with non-trivial destructor}}
    const C &c2 = c0;
    goto *ip; // expected-error {{indirect goto might cross protected scopes}}
  }
}

namespace test13 {
  struct C {
    C(int x);
    ~C();
    int i;
  };
  void f(void **ip) {
    static void *ips[] = { &&l0 };
  l0: // expected-note {{possible target of indirect goto}}
    const int &c1 = C(1).i; // expected-note {{jump exits scope of variable with non-trivial destructor}}
    goto *ip;  // expected-error {{indirect goto might cross protected scopes}}
  }
}

namespace test14 {
  struct C {
    C(int x);
    ~C();
    operator int&() const;
  };
  void f(void **ip) {
    static void *ips[] = { &&l0 };
  l0:
    // no warning since the C temporary is destructed before the goto.
    const int &c1 = C(1);
    goto *ip;
  }
}

// PR14225
namespace test15 {
  void f1() try {
    goto x; // expected-error {{goto into protected scope}}
  } catch(...) {  // expected-note {{jump bypasses initialization of catch block}}
    x: ;
  }
  void f2() try {  // expected-note {{jump bypasses initialization of try block}}
    x: ;
  } catch(...) {
    goto x; // expected-error {{goto into protected scope}}
  }
}
