// RUN: %clang_cc1 -x objective-c -verify -fobjc-arc %s

@interface NSObject

+ (instancetype)new;
+ (instancetype)alloc;

- (void)declaredInSuper;

@end

@interface NSObject (Category)

- (void)declaredInSuperCategory;

@end

@interface Sub: NSObject

- (instancetype)init __attribute__((unavailable)); // expected-note 4 {{'init' has been explicitly marked unavailable here}}

- (void)notImplemented __attribute__((unavailable));

- (void)declaredInSuper __attribute__((unavailable));
- (void)declaredInSuperCategory __attribute__((unavailable));

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
  [self notImplemented];
}

- (void)allowSuperCallUsingSelf {
  [self declaredInSuper];
  [[Sub alloc] declaredInSuper];
  [self declaredInSuperCategory];
  [[Sub alloc] declaredInSuperCategory];
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

void unrelatedContext(void) {
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
