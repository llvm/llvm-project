// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s --implicit-check-not=sanitize_address
// RUN: %clang_cc1 -emit-llvm -o - %s -fsanitize=address | FileCheck %s --check-prefixes=CHECK,ASAN

@interface MyClass
+ (int) addressSafety:(int*)a;
@end

@implementation MyClass

// ASAN: ; Function Attrs:
// ASAN-SAME: sanitize_address
// CHECK-LABEL: define {{.*}}+[MyClass load]
+(void) load { }

// ASAN: ; Function Attrs:
// ASAN-SAME: sanitize_address
// CHECK-LABEL: define {{.*}}+[MyClass addressSafety:]
+ (int) addressSafety:(int*)a { return *a; }

@end
