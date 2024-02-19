// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -fblocks

void tovoid(void*);

void tovoid_test(int (^f)(int, int)) {
  tovoid(f);
}

void reference_lvalue_test(int& (^f)()) {
  f() = 10;
}

// PR 7165
namespace test1 {
  void g(void (^)());
  struct Foo {
    void foo();   
    void test() {
      (void) ^{ foo(); };
    }
  };
}

namespace test2 {
  int repeat(int value, int (^block)(int), unsigned n) {
    while (n--) value = block(value);
    return value;
  }

  class Power {
    int base;

  public:
    Power(int base) : base(base) {}
    int calculate(unsigned n) {
      return repeat(1, ^(int v) { return v * base; }, n);
    }
  };

  int test() {
    return Power(2).calculate(10);
  }
}

namespace radar8382559 {
  void func(bool& outHasProperty);

  int test3() {
    __attribute__((__blocks__(byref))) bool hasProperty = false;
    bool has = true;

    bool (^b)() = ^ {
     func(hasProperty);
     if (hasProperty)
       hasProperty = 0;
     if (has)
       hasProperty = 1;
     return hasProperty;
     };
    func(hasProperty);
    func(has);
    b();
    if (hasProperty)
      hasProperty = 1;
    if (has)
      has = 2;
    return hasProperty = 1;
  }
}

// Move __block variables to the heap when possible.
class MoveOnly {
public:
  MoveOnly();
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&);
};

void move_block() {
  __block MoveOnly mo;
}

// Don't crash after failing to build a block due to a capture of an
// invalid declaration.
namespace test5 {
  struct B { // expected-note 2 {{candidate constructor}}
    void *p;
    B(int); // expected-note {{candidate constructor}}
  };

  void use_block(void (^)());
  void use_block_2(void (^)(), const B &a);

  void test() {
    B x; // expected-error {{no matching constructor for initialization}}
    use_block(^{
        int y;
        use_block_2(^{ (void) y; }, x);
      });
  }
}


// Ensure that we can end function bodies while parsing an
// expression that requires an explicitly-tracked cleanup object
// (i.e. a block literal).

// The nested function body in this test case is a template
// instantiation.  The template function has to be constexpr because
// we'll otherwise delay its instantiation to the end of the
// translation unit.
namespace test6a {
  template <class T> constexpr int func() { return 0; }
  void run(void (^)(), int);

  void test() {
    int aCapturedVar = 0;
    run(^{ (void) aCapturedVar; }, func<int>());
  }
}

// The nested function body in this test case is a method of a local
// class.
namespace test6b {
  void run(void (^)(), void (^)());
  void test() {
    int aCapturedVar = 0;
    run(^{ (void) aCapturedVar; },
        ^{ struct A { static void foo() {} };
            A::foo(); });
  }
}

// The nested function body in this test case is a lambda invocation
// function.
namespace test6c {
  void run(void (^)(), void (^)());
  void test() {
    int aCapturedVar = 0;
    run(^{ (void) aCapturedVar; },
        ^{ struct A { static void foo() {} };
            A::foo(); });
  }
}

namespace test7 {
struct S {};
void f() {
  constexpr S s;
  auto some_block = ^{ (void)s; };
}
}

void static_data_member() {
  auto block = ^{
    class X {
      static int x; // expected-error {{static data member 'x' not allowed in local class 'X'}}
    };
    class Y {
      struct Z {
        static int z; // expected-error {{static data member 'z' not allowed in local struct 'Z'}}
      };
    };
  };
}
