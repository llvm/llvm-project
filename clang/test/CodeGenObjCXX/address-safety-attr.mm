// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck -check-prefix ASAN %s

@interface MyClass
+ (int) addressSafety:(int*)a;
@end

@implementation MyClass

// CHECK-NOT:  +[MyClass load]{{.*}} address_safety
// CHECK:  +[MyClass load]{{.*}}
// ASAN: +[MyClass load]{{.*}} address_safety
+(void) load { }

// CHECK-NOT:  +[MyClass addressSafety:]{{.*}} address_safety
// CHECK:  +[MyClass addressSafety:]{{.*}}
// ASAN:  +[MyClass addressSafety:]{{.*}} address_safety
+ (int) addressSafety:(int*)a { return *a; }

@end
