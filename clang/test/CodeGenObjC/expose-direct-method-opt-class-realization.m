// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple arm64-apple-darwin10 \
// RUN:   -fobjc-expose-direct-methods %s -o - | FileCheck %s

// ============================================================================
// HEURISTIC 1: Classes with +load method skip thunk for class methods
// because they are guaranteed to be realized when the binary is loaded.
// ============================================================================

__attribute__((objc_root_class))
@interface Root
+ (int)rootDirectMethod __attribute__((objc_direct));
@end

@implementation Root

// CHECK-LABEL: define hidden i32 @"+[Root rootDirectMethod]"(ptr noundef %self)
+ (int)rootDirectMethod { return 100; }

@end

@interface ClassWithLoad : Root
+ (void)load;
+ (int)classDirectMethod __attribute__((objc_direct));
@end

@implementation ClassWithLoad

+ (void)load {
  // This method causes the class to be realized at load time
}

// CHECK-LABEL: define hidden i32 @"+[ClassWithLoad classDirectMethod]"(ptr noundef %self)
+ (int)classDirectMethod { return 42; }

@end

// A class without +load method for comparison
@interface ClassWithoutLoad : Root
+ (int)classDirectMethod __attribute__((objc_direct));
@end

@implementation ClassWithoutLoad

// CHECK-LABEL: define hidden i32 @"+[ClassWithoutLoad classDirectMethod]"(ptr noundef %self)
+ (int)classDirectMethod {
  return 42;
}

@end

// CHECK-LABEL: define{{.*}} i32 @testClassWithLoad()
int testClassWithLoad(void) {
  // Because ClassWithLoad has +load, it's guaranteed to be realized.
  // So we should call the implementation directly, NOT through a thunk.
  //
  // CHECK: call i32 @"+[ClassWithLoad classDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[ClassWithLoad classDirectMethod]_thunk"
  return [ClassWithLoad classDirectMethod];
}

// CHECK-LABEL: define{{.*}} i32 @testClassWithoutLoad()
int testClassWithoutLoad(void) {
  // ClassWithoutLoad has no +load, so the class might not be realized.
  // We need to call through the thunk which will realize the class.
  //
  // CHECK: call i32 @"+[ClassWithoutLoad classDirectMethod]_thunk"(ptr noundef
  return [ClassWithoutLoad classDirectMethod];
}

// ============================================================================
// HEURISTIC 2: Calls from within the same class skip thunk
// because if we're executing a method of the class, it must be realized.
// ============================================================================

@interface SameClassTest : Root
+ (int)classDirectMethod __attribute__((objc_direct));
+ (int)callerClassMethod __attribute__((objc_direct));
- (int)callerInstanceMethod __attribute__((objc_direct));
@end

@implementation SameClassTest

// CHECK-LABEL: define hidden i32 @"+[SameClassTest classDirectMethod]"(ptr noundef %self)
+ (int)classDirectMethod {
  return 42;
}

// CHECK-LABEL: define hidden i32 @"+[SameClassTest callerClassMethod]"(ptr noundef %self)
+ (int)callerClassMethod {
  // Calling a class method from another class method of the SAME class.
  // The class must be realized (we're already executing a method of it).
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[SameClassTest classDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[SameClassTest classDirectMethod]_thunk"
  int a = [SameClassTest classDirectMethod];

  // Calling the root class's class method from a subclass method.
  // Root must be realized because SubClass inherits from it.
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[Root rootDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[Root rootDirectMethod]_thunk"
  int b = [Root rootDirectMethod];

  return a + b;
}

// CHECK-LABEL: define hidden i32 @"-[SameClassTest callerInstanceMethod]"(ptr noundef %self)
- (int)callerInstanceMethod {
  // Calling a class method from an instance method of the SAME class.
  // The class must be realized (we're already executing a method of it).
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[SameClassTest classDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[SameClassTest classDirectMethod]_thunk"
  int a = [SameClassTest classDirectMethod];

  // Calling the root class's class method from a subclass instance method.
  // Root must be realized because SubClass inherits from it.
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[Root rootDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[Root rootDirectMethod]_thunk"
  int b = [Root rootDirectMethod];

  return a + b;
}

@end

// ============================================================================
// HEURISTIC 3: Previously realized classes in the same basic block skip thunk.
// If we've already called a class method (which realizes the class),
// subsequent calls to the same class or its superclasses can skip the thunk.
// ============================================================================

// CHECK-LABEL: define{{.*}} i32 @testPreviouslyRealizedParentClass
int testPreviouslyRealizedParentClass(int flag) {
  if (flag) {
    // First call to ClassWithoutLoad - needs thunk (class might not be realized)
    // CHECK: call i32 @"+[ClassWithoutLoad classDirectMethod]_thunk"(ptr noundef
    int a = [ClassWithoutLoad classDirectMethod];

    // Second call to same class - should skip thunk (class was just realized)
    // CHECK: call i32 @"+[ClassWithoutLoad classDirectMethod]"(ptr noundef
    // CHECK-NOT: call i32 @"+[ClassWithoutLoad classDirectMethod]_thunk"
    int b = [ClassWithoutLoad classDirectMethod];

    // Call to Root (parent of ClassWithoutLoad) - should skip thunk
    // because realizing ClassWithoutLoad also realizes its superclass Root.
    // CHECK: call i32 @"+[Root rootDirectMethod]"(ptr noundef
    // CHECK-NOT: call i32 @"+[Root rootDirectMethod]_thunk"
    int c = [Root rootDirectMethod];
    return a + b + c;

  }
  // New block, we are not sure if prev block is executed, so we have to conservatively realize again.
  // CHECK: call i32 @"+[ClassWithoutLoad classDirectMethod]_thunk"
  // CHECK-NOT: call i32 @"+[ClassWithoutLoad classDirectMethod]"(ptr noundef
  int b = [ClassWithoutLoad classDirectMethod];
  // CHECK: call i32 @"+[Root rootDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[Root rootDirectMethod]_thunk"
  int c = [Root rootDirectMethod];

  return b + c;
}
