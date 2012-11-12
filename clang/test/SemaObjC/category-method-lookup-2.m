// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef struct objc_class *Class;
@interface NSObject
- (Class)class;
@end
@interface Bar : NSObject
@end
@interface Bar (Cat)
@end

// NOTE: No class implementation for Bar precedes this category definition.
@implementation Bar (Cat)

// private method.
+ classMethod { return self; }

- instanceMethod {
  [[self class] classMethod];
  return 0;
}

@end
