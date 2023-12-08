// RUN: %check_clang_tidy %s objc-super-self %t

@interface NSObject
- (instancetype)init;
- (instancetype)self;
@end

@interface NSObjectDerivedClass : NSObject
@end

@implementation NSObjectDerivedClass

- (instancetype)init {
  return [super self];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious invocation of 'self' in initializer; did you mean to invoke a superclass initializer? [objc-super-self]
// CHECK-FIXES: return [super init];
}

- (instancetype)initWithObject:(NSObject *)obj {
  self = [super self];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious invocation of 'self' in initializer; did you mean to invoke a superclass initializer? [objc-super-self]
// CHECK-FIXES: self = [super init];
  if (self) {
    // ...
  }
  return self;
}

#define INITIALIZE() [super self]

- (instancetype)initWithObject:(NSObject *)objc a:(int)a {
  return INITIALIZE();
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious invocation of 'self' in initializer; did you mean to invoke a superclass initializer? [objc-super-self]
// CHECK-FIXES: return INITIALIZE();
}

#define INITIALIZER_IMPL() return [super self]

- (instancetype)initWithObject:(NSObject *)objc b:(int)b {
  INITIALIZER_IMPL();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: suspicious invocation of 'self' in initializer; did you mean to invoke a superclass initializer? [objc-super-self]
// CHECK-FIXES: INITIALIZER_IMPL();
}

#define INITIALIZER_METHOD self

- (instancetype)initWithObject:(NSObject *)objc c:(int)c {
  return [super INITIALIZER_METHOD];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious invocation of 'self' in initializer; did you mean to invoke a superclass initializer? [objc-super-self]
// CHECK-FIXES: return [super INITIALIZER_METHOD];
}

#define RECEIVER super

- (instancetype)initWithObject:(NSObject *)objc d:(int)d {
  return [RECEIVER self];
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious invocation of 'self' in initializer; did you mean to invoke a superclass initializer? [objc-super-self]
// CHECK-FIXES: return [RECEIVER self];
}

- (instancetype)foo {
  return [super self];
}

- (instancetype)bar {
  return [self self];
}

@end

@interface RootClass
- (instancetype)init;
- (instancetype)self;
@end

@interface NotNSObjectDerivedClass : RootClass
@end

@implementation NotNSObjectDerivedClass

- (instancetype)init {
  return [super self];
}

@end

