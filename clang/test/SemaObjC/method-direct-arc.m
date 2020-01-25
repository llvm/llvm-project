// RUN: %clang_cc1 -fobjc-arc -fsyntax-only -verify -Wselector-type-mismatch %s

extern Class object_getClass(id);

__attribute__((objc_root_class))
@interface Root
- (Class)class;
+ (void)directMethod __attribute__((objc_direct)); // expected-note {{direct method 'directMethod' declared here}}
+ (void)anotherDirectMethod __attribute__((objc_direct));
@end

@implementation Root
- (Class)class
{
  return object_getClass(self);
}
+ (void)directMethod {
}
+ (void)anotherDirectMethod {
  [self directMethod]; // this should not warn
}
+ (void)regularMethod {
  [self directMethod];        // this should not warn
  [self anotherDirectMethod]; // this should not warn
}
- (void)regularInstanceMethod {
  [[self class] directMethod]; // expected-error {{messaging a Class with a method that is possibly direct}}
}
@end

@interface Sub : Root
@end

@implementation Sub
+ (void)foo {
  [self directMethod]; // this should not warn
}
@end

__attribute__((objc_root_class))
@interface Other
@end

@implementation Other
+ (void)bar {
  [self directMethod]; // expected-error {{no known class method for selector 'directMethod'}}
}
@end
