// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-X86-64 %s

__attribute__((weak_import)) @interface WeakRootClass @end

__attribute__((weak_import)) @interface WeakClass : WeakRootClass
@end

@interface MySubclass : WeakClass @end

@implementation MySubclass @end

@implementation WeakClass(MyCategory) @end


__attribute__((weak_import))
@interface WeakClass1 @end

@implementation WeakClass1(MyCategory) @end

@implementation WeakClass1(YourCategory) @end

 __attribute__((weak_import))
@interface WeakClass3 
+ message;
@end

int main(void) {
     [WeakClass3 message];
}

// CHECK-X86-64: OBJC_METACLASS_$_WeakRootClass" = extern_weak global
// CHECK-X86-64: OBJC_METACLASS_$_WeakClass" = extern_weak global
// CHECK-X86-64: OBJC_CLASS_$_WeakClass" = extern_weak global
// CHECK-X86-64: OBJC_CLASS_$_WeakClass1" = extern_weak global
// CHECK-X86-64: OBJC_CLASS_$_WeakClass3" = extern_weak global

// Root is being implemented here. No extern_weak.
__attribute__((weak_import)) @interface Root @end

@interface Super : Root @end

@interface Sub : Super @end

@implementation Sub @end

@implementation Root @end

// CHECK-X86-64-NOT: OBJC_METACLASS_$_Root" = extern_weak global
