// RUN: %clang_cc1 -x objective-c++ -std=c++11 -fobjc-arc -fblocks -Wimplicit-retain-self -verify %s
// rdar://11194874

typedef void (^BlockTy)();

void noescapeFunc(__attribute__((noescape)) BlockTy);
void escapeFunc(BlockTy);

@interface Root @end

@interface I : Root
{
  int _bar;
}
@end

@implementation I
  - (void)foo{
      ^{
           _bar = 3; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
       }();
  }

  - (void)testNested{
    noescapeFunc(^{
      (void)_bar;
      escapeFunc(^{
        (void)_bar; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
        noescapeFunc(^{
          (void)_bar; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
        });
        (void)_bar; // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
      });
      (void)_bar;
    });
  }

  - (void)testLambdaInBlock{
    noescapeFunc(^{ [&](){ (void)_bar; }(); });
    escapeFunc(^{ [&](){ (void)_bar; }(); }); // expected-warning {{block implicitly retains 'self'; explicitly mention 'self' to indicate this is intended behavior}}
  }

  - (BlockTy)testDeclType{
    return ^{ decltype(_bar) i = 12; (void)i; };
  }
@end
