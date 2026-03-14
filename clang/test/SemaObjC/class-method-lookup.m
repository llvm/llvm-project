// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface MyBase 
- (void) rootInstanceMethod;
@end

@interface MyIntermediate: MyBase
@end

@interface MyDerived: MyIntermediate
- (void) instanceMethod;
+ (void) classMethod;
@end

@implementation MyDerived
- (void) instanceMethod {
}

+ (void) classMethod {                    /* If a class method is not found, the root  */
    [self rootInstanceMethod];            /* class is searched for an instance method  */
    [MyIntermediate rootInstanceMethod];  /* with the same name.                       */

    [self instanceMethod];// expected-warning {{'+instanceMethod' not found (return type defaults to 'id')}}
    [MyDerived instanceMethod];// expected-warning {{'+instanceMethod' not found (return type defaults to 'id')}}
}
@end

@interface Object @end

@interface Class1
- (void)setWindow:(Object *)wdw;
@end

@interface Class2
- (void)setWindow:(Class1 *)window;
@end

#define nil (void*)0

id foo(void) {
  Object *obj;
  id obj2 = obj;
  [obj setWindow:nil]; // expected-warning {{'Object' may not respond to 'setWindow:'}}

  return obj;
}
