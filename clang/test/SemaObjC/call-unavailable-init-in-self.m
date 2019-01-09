// RUN: %clang_cc1 -x objective-c -verify -fobjc-arc %s

@interface NSObject

+ (instancetype)new;
+ (instancetype)alloc;

@end

@interface Sub: NSObject

- (instancetype)init __attribute__((unavailable)); // expected-note 4 {{'init' has been explicitly marked unavailable here}}

- (void)notImplemented __attribute__((unavailable)); // expected-note {{'notImplemented' has been explicitly marked unavailable here}}

@end

@implementation Sub

+ (Sub *)create {
  return [[self alloc] init];
}

+ (Sub *)create2 {
  return [self new];
}

+ (Sub *)create3 {
  return [Sub new];
}

- (instancetype) init {
  return self;
}

- (void)reportUseOfUnimplemented {
  [self notImplemented]; // expected-error {{'notImplemented' is unavailable}}
}

@end

@interface SubClassContext: Sub
@end

@implementation SubClassContext

- (void)subClassContext {
  (void)[[Sub alloc] init]; // expected-error {{'init' is unavailable}}
  (void)[Sub new]; // expected-error {{'new' is unavailable}}
}

@end

void unrelatedContext() {
  (void)[[Sub alloc] init]; // expected-error {{'init' is unavailable}}
  (void)[Sub new]; // expected-error {{'new' is unavailable}}
}

@interface X @end

@interface X (Foo)
-(void)meth __attribute__((unavailable));
@end

@implementation X (Foo)
-(void)meth {}
-(void)call_it { [self meth]; }
@end
