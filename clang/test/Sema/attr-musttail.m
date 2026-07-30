// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-objc-root-class -verify %s

void TestObjcBlock(void) {
  void (^x)(void) = ^(void) {
    __attribute__((musttail)) return TestObjcBlock(); // expected-error{{'musttail' attribute cannot be used from a block}}
  };
  __attribute__((musttail)) return x();
}

void ReturnsVoid(void);
void TestObjcBlockVar(void) {
  __block int i = 0;                              // expected-note{{jump exits scope of __block variable}}
  __attribute__((musttail)) return ReturnsVoid(); // expected-error{{cannot perform a tail call from this return statement}}
}

__attribute__((objc_root_class))
@interface TestObjcClass
@end

@implementation TestObjcClass

- (void)testObjCMethod {
  __attribute__((musttail)) return ReturnsVoid(); // expected-error{{'musttail' attribute cannot be used from an Objective-C function}}
}

@end
