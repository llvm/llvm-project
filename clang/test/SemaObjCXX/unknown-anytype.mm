// RUN: %clang_cc1 -fdebugger-support -funknown-anytype -fsyntax-only -verify %s

// rdar://problem/9416370
namespace test0 {
  void test(id x) {
    if ([x foo]) {} // expected-error {{no known method '-foo'; cast the message send to the method's return type}}
    [x foo]; // expected-error {{no known method '-foo'; cast the message send to the method's return type}}
  }
}

// rdar://problem/12565338
@interface Test1
- (void) test_a: (__unknown_anytype)foo;
- (void) test_b: (__unknown_anytype)foo;
- (void) test_c: (__unknown_anytype)foo;
@end
namespace test1 {
  struct POD {
    int x;
  };

  void a(Test1 *obj) {
    POD v;
    [obj test_a: v];
  }

  struct Uncopyable {
    Uncopyable();
  private:
    Uncopyable(const Uncopyable &); // expected-note {{declared private here}}
  };

  void b(Test1 *obj) {
    Uncopyable v;
    [obj test_b: v]; // expected-error {{calling a private constructor}}
  }

  void c(Test1 *obj) {
    Uncopyable v;
    [obj test_c: (const Uncopyable&) v];
  }
}

// Just test that we can declare a function taking __unknown_anytype.
// For now, we don't actually need to make calling something like this
// work; if that changes, here's what's required:
//   - get this call through overload resolution somehow,
//   - update the function-call argument-passing code like the
//     message-send code, and
//   - rewrite the function expression to have a type that doesn't
//     involving __unknown_anytype.
namespace test2 {
  void foo(__unknown_anytype x);
}
