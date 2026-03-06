// Test variadic direct methods - use inline preconditions instead of thunks
// to avoid musttail complexity across different architectures.
// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple arm64-apple-darwin10 \
// RUN:   -fobjc-direct-precondition-thunk %s -o - | FileCheck %s

__attribute__((objc_root_class))
@interface Root
- (int)varMethod:(int)first, ... __attribute__((objc_direct));
+ (void)printf:(Root *)format, ... __attribute__((objc_direct));
@end

// Add a weakly linked class and a weakly linked class method
__attribute__((objc_root_class, weak_import))
@interface WeakRoot
+ (int)weakPrintf:(int)first, ... __attribute__((objc_direct, weak_import));
@end


@implementation Root

// Variadic methods use inline preconditions instead of thunks
// to avoid musttail complexity across different architectures.
// CHECK-LABEL: define hidden i32 @"-[Root varMethod:]"(
// CHECK-NOT: @"\01-[Root varMethod:]"
// CHECK-NOT: @"-[Root varMethod:]_thunk"
- (int)varMethod:(int)first, ... {
  // Should NOT have nil check (moved to caller)
  // CHECK-NOT: icmp eq ptr {{.*}}, null
  // CHECK-NOT: objc_direct_method.self_is_nil
  return first;
}

// CHECK-LABEL: define hidden void @"+[Root printf:]"(
// CHECK-NOT: @"\01+[Root printf:]"
// CHECK-NOT: @"+[Root printf:]_thunk"
// Class realization should be at the call site, not in the implementation.
// CHECK-NOT: objc_msgSend
+ (void)printf:(Root *)format, ... {}

@end

// Test: Nullable receiver should have inline nil check
// CHECK-LABEL: define{{.*}} void @useRoot(
void useRoot(Root *_Nullable root) {
  // For nullable receivers, we should emit nil check inline
  // CHECK: icmp eq ptr %{{[0-9]+}}, null
  // CHECK: br i1 %{{[0-9]+}}, label %msgSend.null-receiver, label %msgSend.call

  // CHECK: msgSend.call:
  // CHECK: call i32 (ptr, i32, ...) @"-[Root varMethod:]"(ptr noundef %{{[0-9]+}}, i32 noundef 1, i32 noundef 2, double noundef 3.0{{.*}})
  // CHECK: br label %msgSend.cont

  // CHECK: msgSend.null-receiver:
  // CHECK: br label %msgSend.cont

  // CHECK: msgSend.cont:
  [root varMethod:1, 2, 3.0];

  // Class realization before call
  // CHECK: %{{.*}} = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$
  // CHECK: %{{.*}} = load ptr, ptr @OBJC_SELECTOR_REFERENCES_,
  // CHECK: %{{.*}} = call ptr @objc_msgSend
  // CHECK: call void (ptr, ptr, ...) @"+[Root printf:]"(
  [Root printf:root, "hello", root];

  // For weakly linked class, inline realization first
  // CHECK: %{{.*}} = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$
  // CHECK: %{{.*}} = load ptr, ptr @OBJC_SELECTOR_REFERENCES_,
  // CHECK: %{{.*}} = call ptr @objc_msgSend

  // Then perform nil check
  // CHECK: %{{.*}} = icmp eq ptr %{{.*}}, null
  // CHECK: br i1 %{{.*}}, label %msgSend.null-receiver{{.*}}, label %msgSend.call{{.*}}

  // Finally call the class method
  // CHECK: %{{.*}} = call i32 (ptr, i32, ...) @"+[WeakRoot weakPrintf:]"
  [WeakRoot weakPrintf: 1, 2, 3.0];
}

// Test: Non-null receiver

// CHECK-LABEL: define{{.*}} void @useRootNonNull(
void useRootNonNull(Root *_Nonnull root) {
  // TODO: For nonnull receivers, we should not emit nil check inline
  // CHECK: icmp eq ptr %{{[0-9]+}}, null
  // CHECK: br i1 %{{[0-9]+}}, label %msgSend.null-receiver, label %msgSend.call

  // CHECK: msgSend.call:
  // CHECK: call i32 (ptr, i32, ...) @"-[Root varMethod:]"(ptr noundef %{{[0-9]+}}, i32 noundef 1, i32 noundef 2, double noundef 3.0{{.*}})
  // CHECK: br label %msgSend.cont

  // CHECK: msgSend.null-receiver:
  // CHECK: br label %msgSend.cont

  // CHECK: msgSend.cont:
  [root varMethod:1, 2, 3.0];

  // Class realization before call
  // CHECK: %{{.*}} = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$
  // CHECK: %{{.*}} = load ptr, ptr @OBJC_SELECTOR_REFERENCES_,
  // CHECK: %{{.*}} = call ptr @objc_msgSend
  // CHECK: call void (ptr, ptr, ...) @"+[Root printf:]"(
  [Root printf:root, "hello", root];
}
