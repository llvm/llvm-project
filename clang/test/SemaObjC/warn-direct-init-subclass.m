// RUN: %clang_cc1 -fsyntax-only -Wno-objc-designated-initializers -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-objc-designated-initializers -fobjc-arc -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-objc-designated-initializers -x objective-c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-objc-designated-initializers -Wno-objc-direct-init-subclass -verify=off %s

// off-no-diagnostics

#define DIRECT __attribute__((objc_direct))
#define NS_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))

__attribute__((objc_root_class))
@interface Root
+ (instancetype)alloc;
@end

@interface Parent : Root
- (instancetype)initWithParent:(int)x DIRECT NS_DESIGNATED_INITIALIZER; // #Parent_initWithParent
- (instancetype)initWithValue:(int)x NS_DESIGNATED_INITIALIZER;
// Parent marks its designated initializers, so this one is a convenience
// initializer.
- (instancetype)initConvenience DIRECT;
+ (instancetype)makeParent;
@end

// Also a convenience initializer, since Parent marks its designated ones.
@interface Parent (Cat)
- (instancetype)initFromCategory:(int)x DIRECT;
@end

@interface Child : Parent
- (instancetype)initWithTag:(int)tag NS_DESIGNATED_INITIALIZER;
- (instancetype)initConvenienceChild;
+ (instancetype)make;
@end

@interface GrandChild : Child
@end

@implementation Parent
- (instancetype)initWithParent:(int)x {
  return self;
}
- (instancetype)initWithValue:(int)x {
  return self;
}
- (instancetype)initConvenience {
  // The receiver's static type is Parent.
  return [self initWithParent:0];
}
+ (instancetype)makeParent {
  // Class values aren't tracked.
  return [[self alloc] initWithParent:0];
}
@end

@implementation Child
- (instancetype)initWithTag:(int)tag {
  // A designated initializer chains up through self, since
  // [super initWithParent:] is an error for a direct method.
  self = [self initWithParent:tag];
  return self;
}
- (instancetype)initConvenienceChild {
  return [self initWithParent:0]; // expected-warning {{direct initializer 'initWithParent:' of 'Parent' leaves subclass 'Child' partially initialized}}
                                  // expected-note@#Parent_initWithParent {{direct method 'initWithParent:' declared here}}
}
+ (instancetype)make {
  return [[self alloc] initWithParent:0]; // expected-warning {{direct initializer 'initWithParent:' of 'Parent' leaves subclass 'Child' partially initialized}}
                                          // expected-note@#Parent_initWithParent {{direct method 'initWithParent:' declared here}}
}
@end

void test(Child *child) {
  (void)[[Child alloc] initWithParent:1]; // expected-warning {{direct initializer 'initWithParent:' of 'Parent' leaves subclass 'Child' partially initialized}}
                                          // expected-note@#Parent_initWithParent {{direct method 'initWithParent:' declared here}}
  (void)[child initWithParent:1]; // expected-warning {{direct initializer 'initWithParent:' of 'Parent' leaves subclass 'Child' partially initialized}}
                                  // expected-note@#Parent_initWithParent {{direct method 'initWithParent:' declared here}}
  (void)[[GrandChild alloc] initWithParent:1]; // expected-warning {{direct initializer 'initWithParent:' of 'Parent' leaves subclass 'GrandChild' partially initialized}}
                                               // expected-note@#Parent_initWithParent {{direct method 'initWithParent:' declared here}}

  (void)[[Parent alloc] initWithParent:1];
  // A cast says the caller means it.
  (void)[(Parent *)[Child alloc] initWithParent:1];
  (void)[[Child alloc] initConvenience];
  (void)[[Child alloc] initFromCategory:1];
  (void)[[Child alloc] initWithValue:1];
  // Parent's factory runs initWithParent: on a Child, but factories aren't
  // checked.
  (void)[Child makeParent];
}

// Plain marks no designated initializers, so all of its initializers are
// designated, including the ones in categories.
@interface Plain : Root
- (instancetype)initWithPlain:(int)x DIRECT; // #Plain_initWithPlain
@end

@interface Plain (Cat)
- (instancetype)initFromPlainCategory:(int)x DIRECT; // #Plain_initFromPlainCategory
@end

@interface PlainChild : Plain
@end

void testPlain(void) {
  (void)[[PlainChild alloc] initWithPlain:1]; // expected-warning {{direct initializer 'initWithPlain:' of 'Plain' leaves subclass 'PlainChild' partially initialized}}
                                              // expected-note@#Plain_initWithPlain {{direct method 'initWithPlain:' declared here}}
  (void)[[PlainChild alloc] initFromPlainCategory:1]; // expected-warning {{direct initializer 'initFromPlainCategory:' of 'Plain' leaves subclass 'PlainChild' partially initialized}}
                                                      // expected-note@#Plain_initFromPlainCategory {{direct method 'initFromPlainCategory:' declared here}}
}
