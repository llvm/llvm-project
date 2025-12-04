// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple arm64-apple-darwin10 \
// RUN:   -fobjc-expose-direct-methods %s -o - | FileCheck %s

// ============================================================================
// HEURISTIC 1: Classes with +load method skip thunk for class methods
// because they are guaranteed to be realized when the binary is loaded.
// ============================================================================

__attribute__((objc_root_class))
@interface ClassWithLoad
+ (void)load;
+ (int)classDirectMethod __attribute__((objc_direct));
@end

@implementation ClassWithLoad

+ (void)load {
  // This method causes the class to be realized at load time
}

// CHECK-LABEL: define hidden i32 @"+[ClassWithLoad classDirectMethod]"(ptr noundef %self)
+ (int)classDirectMethod {
  return 42;
}

@end

// A class without +load method for comparison
__attribute__((objc_root_class))
@interface ClassWithoutLoad
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

__attribute__((objc_root_class))
@interface SameClassTest
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
  return [SameClassTest classDirectMethod];
}

// CHECK-LABEL: define hidden i32 @"-[SameClassTest callerInstanceMethod]"(ptr noundef %self)
- (int)callerInstanceMethod {
  // Calling a class method from an instance method of the SAME class.
  // The class must be realized (we're already executing a method of it).
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[SameClassTest classDirectMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[SameClassTest classDirectMethod]_thunk"
  return [SameClassTest classDirectMethod];
}

@end

__attribute__((objc_root_class))
@interface SuperClass
+ (int)superClassMethod __attribute__((objc_direct));
@end

@implementation SuperClass

// CHECK-LABEL: define hidden i32 @"+[SuperClass superClassMethod]"(ptr noundef %self)
+ (int)superClassMethod {
  return 100;
}

@end

@interface SubClass : SuperClass
+ (int)subCallerMethod __attribute__((objc_direct));
- (int)subInstanceCaller __attribute__((objc_direct));
@end

@implementation SubClass

// CHECK-LABEL: define hidden i32 @"+[SubClass subCallerMethod]"(ptr noundef %self)
+ (int)subCallerMethod {
  // Calling a superclass's class method from a subclass method.
  // SuperClass must be realized because SubClass inherits from it.
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[SuperClass superClassMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[SuperClass superClassMethod]_thunk"
  return [SuperClass superClassMethod];
}

// CHECK-LABEL: define hidden i32 @"-[SubClass subInstanceCaller]"(ptr noundef %self)
- (int)subInstanceCaller {
  // Calling a superclass's class method from a subclass instance method.
  // SuperClass must be realized because SubClass inherits from it.
  // Should call implementation directly, NOT through thunk.
  //
  // CHECK: call i32 @"+[SuperClass superClassMethod]"(ptr noundef
  // CHECK-NOT: call i32 @"+[SuperClass superClassMethod]_thunk"
  return [SuperClass superClassMethod];
}

@end
