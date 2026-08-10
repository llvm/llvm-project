// RUN: %clang_cc1 %s -fblocks -verify -fsyntax-only

@interface NSObject
@end

@interface Test : NSObject
- (void)test;
@end

@implementation Test
- (void)test
{
  void (^simpleBlock)(void) = ^ _Nonnull { //expected-warning {{attribute '_Nonnull' ignored, because it cannot be applied to omitted return type}}
    return;
  };
  void (^simpleBlock2)(void) = ^ _Nonnull void { //expected-error {{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'void'}}
    return;
  };
  void (^simpleBlock3)(void) = ^ _Nonnull (void) {  //expected-warning {{attribute '_Nonnull' ignored, because it cannot be applied to omitted return type}}
    return;
  };

  void (^simpleBlock4)(void) = ^ const { //expected-warning {{'const' qualifier on omitted return type '<dependent type>' has no effect}}
    return;
  };
  void (^simpleBlock5)(void) = ^ const void { //expected-error {{incompatible block pointer types initializing 'void (^)(void)' with an expression of type 'const void (^)(void)'}}
    return; // expected-warning@-1 {{function cannot return qualified void type 'const void'}}
  };
  void (^simpleBlock6)(void) = ^ const (void) { //expected-warning {{'const' qualifier on omitted return type '<dependent type>' has no effect}}
    return;
  };
  void (^simpleBlock7)(void) = ^ _Nonnull __attribute__((align_value(128))) _Nullable const (void) { // expected-warning {{attribute '_Nullable' ignored, because it cannot be applied to omitted return type}} \
    // expected-warning {{attribute '_Nonnull' ignored, because it cannot be applied to omitted return type}} \
    // expected-warning {{'const' qualifier on omitted return type '<dependent type>' has no effect}} \
    // expected-warning {{'align_value' attribute only applies to variables and typedefs}}
    return;
  };
  void (^simpleBlock9)(void) = ^ __attribute__ ((align_value(128))) _Nonnull const (void) { // expected-warning {{attribute '_Nonnull' ignored, because it cannot be applied to omitted return type}} \
    // expected-warning {{'const' qualifier on omitted return type '<dependent type>' has no effect}} \
    // expected-warning {{'align_value' attribute only applies to variables and typedefs}}
    return;
  };
}
@end
