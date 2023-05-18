// RUN: %clang_cc1 -Wobjc-multiple-method-names -x objective-c -verify %s
// RUN: %clang_cc1 -Wobjc-multiple-method-names -x objective-c -verify -fobjc-arc %s
// expected-no-diagnostics

@interface NSObj

+ (instancetype) alloc;

+ (_Nonnull instancetype) globalObject;

@end

@interface SelfAllocReturn: NSObj

- (instancetype)initWithFoo:(int)x;

@end

@interface SelfAllocReturn2: NSObj

- (instancetype)initWithFoo:(SelfAllocReturn *)x;

@end

@implementation SelfAllocReturn

- (instancetype)initWithFoo:(int)x {
    return self;
}

+ (instancetype) thingWithFoo:(int)x {
    return [[self alloc] initWithFoo: x];
}

+ (void) initGlobal {
  (void)[[self globalObject] initWithFoo: 20];
}

@end
