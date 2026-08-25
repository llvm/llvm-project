// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fobjc-arc -fblocks -Wblocks-capturing-this -Wblocks-capturing-reference -Wblocks-capturing-raw-pointer -Wimplicit-retain-self -verify %s

typedef void (^BlockTy)();

void noescapeFunc(__attribute__((noescape)) BlockTy);
void escapeFunc(BlockTy);

@interface Root @end

@interface I : Root
- (void)foo;
@end

class F {
 public:
  void Foo() const;
  void FooAsync(F* p, F& r, I* i) {
    escapeFunc(
        ^{
          Foo(); // expected-warning {{block implicitly captures 'this'}}
          p->Foo(); // expected-warning {{block implicitly captures a raw pointer}}
          r.Foo(); // expected-warning {{block implicitly captures a C++ reference}}
          [i foo];
        });

    ([=, &r]() {
      escapeFunc(
          ^{
            Foo(); // expected-warning {{block implicitly captures 'this'}}
            p->Foo(); // expected-warning {{block implicitly captures a raw pointer}}
            r.Foo(); // expected-warning {{block implicitly captures a C++ reference}}
            [i foo];
          });
    })();

    escapeFunc(
        ^{
          ([=]() {
            Foo(); // expected-warning {{block implicitly captures 'this'}}
            p->Foo(); // expected-warning {{block implicitly captures a raw pointer}}
            r.Foo();  // expected-warning {{block implicitly captures a C++ reference}}
            [i foo];
          })();
        });

    noescapeFunc(
        ^{
          Foo();
          p->Foo();
          r.Foo();
          [i foo];
        });
  }
};

@implementation I {
  int _bar;
}

- (void)doSomethingWithPtr:(F*)p
                       ref:(F&)r
                       obj:(I*)i {
    escapeFunc(
        ^{
          (void)_bar; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
          p->Foo(); // expected-warning {{block implicitly captures a raw pointer}}
          r.Foo(); // expected-warning {{block implicitly captures a C++ reference}}
          [i foo];
        });

    ([=, &r]() {
      escapeFunc(
          ^{
            (void)_bar; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
            p->Foo(); // expected-warning {{block implicitly captures a raw pointer}}
            r.Foo(); // expected-warning {{block implicitly captures a C++ reference}}
            [i foo];
          });
    })();

    escapeFunc(
        ^{
          ([=]() {
            (void)_bar; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
            p->Foo(); // expected-warning {{block implicitly captures a raw pointer}}
            r.Foo(); // expected-warning {{block implicitly captures a C++ reference}}
            [i foo];
          })();
        });

    noescapeFunc(
        ^{
          (void)_bar;
          p->Foo();
          r.Foo();
          [i foo];
        });
}

- (void)foo {
}

@end
