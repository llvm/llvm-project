
@interface Object

- (int)instanceMethod;

@end

@interface Wrapper

- (Object *)returnsObject:(int)arg;

- (Object *)classMethodReturnsObject;
+ (Object *)classMethodReturnsObject;

@end

void differentWrapperVariables(Wrapper *wrapper) {
  [[wrapper returnsObject: 42] instanceMethod];
  Wrapper *copyWrapper = wrapper;
  if (wrapper) {
    Wrapper *wrapper = copyWrapper;
    [[wrapper returnsObject: 42] prop];
  }
  [[Wrapper classMethodReturnsObject] instanceMethod];
  if (wrapper) {
    __auto_type Wrapper = wrapper;
    [[Wrapper classMethodReturnsObject] instanceMethod];
  }
}

// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:18:1-48 -in=%s:24:1-55 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action!
