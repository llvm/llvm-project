// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface NSObject {}

@end

@interface MyClass : NSObject {}

@end

@interface MyClass (MyCategorie)

@end

@interface MySubClass : MyClass {}

@end

@interface MySubSubClass : MySubClass {}

@end

@implementation NSObject (NSObjectCategory)
- (void)rootMethod {}
@end

@implementation MyClass

+ (void)myClassMethod { }
- (void)myMethod { }

@end

@implementation MyClass (MyCategorie)
+ (void)myClassCategoryMethod { }
- (void)categoryMethod {}
@end

@implementation MySubClass

- (void)mySubMethod {}

- (void)myTest {
  [self mySubMethod];
  // should lookup method in superclass implementation if available
  [self myMethod];
  [super myMethod];
  
  [self categoryMethod];
  [super categoryMethod];
  
  // instance method of root class
  [MyClass rootMethod];
  
  [MyClass myClassMethod];
  [MySubClass myClassMethod];
  
  [MyClass myClassCategoryMethod];
  [MySubClass myClassCategoryMethod];
}

@end
