// RUN: rm -rf %t
// RUN: split-file %s %t

// Test 1: Check IR from library implementation (visibility attributes)
// RUN: %clang_cc1 -triple arm64-apple-darwin -fobjc-arc \
// RUN:   -fobjc-direct-precondition-thunk -emit-llvm -o - %t/foo.m \
// RUN:   -I %t | FileCheck %s --check-prefix=FOO_M

// Test 2: Check IR from main (consumer)
// RUN: %clang_cc1 -triple arm64-apple-darwin -fobjc-arc \
// RUN:   -fobjc-direct-precondition-thunk -emit-llvm -o - %t/main.m \
// RUN:   -I %t | FileCheck %s --check-prefix=MAIN_M

//--- foo.h
// Header for libFoo
__attribute__((objc_root_class))
@interface Foo
// Direct properties with default hidden visibility
@property (nonatomic, direct) int privateValue;

// Direct properties with explicit default visibility
@property (nonatomic, direct) int exportedValue __attribute__((visibility("default")));

+ (instancetype)alloc;
- (instancetype)initWithprivateValue:(int)x exportedValue:(int)y;
// Default hidden visibility
- (int)instanceMethod:(int)x __attribute__((objc_direct));
+ (int)classMethod:(int)x __attribute__((objc_direct));

// Explicit default visibility (should be exported)
- (int)exportedInstanceMethod:(int)x __attribute__((objc_direct, visibility("default")));
+ (int)exportedClassMethod:(int)x __attribute__((objc_direct, visibility("default")));
@end

//--- foo.m

// libFoo does not have thunks because the true implementation is not used internally.
// FOO_M-NOT: @{{.*}}_thunk
#import "foo.h"

@implementation Foo

// FOO_M-LABEL: define internal ptr @"\01-[Foo initWithprivateValue:exportedValue:]"
- (instancetype)initWithprivateValue:(int)x exportedValue:(int)y {
    _privateValue = x;
    _exportedValue = y;
  return self;
}

// FOO_M-LABEL: define hidden i32 @"-[Foo instanceMethod:]D"
- (int)instanceMethod:(int)x {
  // Compiler is smart enough to know that self is non-nil, so we dispatch to
  // true implementation.
  // FOO_M: call i32 @"-[Foo privateValue]D"
  // FOO_M: call i32 @"-[Foo exportedValue]D"
  return x + [self privateValue] + [self exportedValue];
}

// Hidden property getter and setter (default visibility)
// FOO_M-LABEL: define hidden i32 @"-[Foo privateValue]D"

// Exported property getter and setter (explicit default visibility)
// FOO_M-LABEL: define dso_local i32 @"-[Foo exportedValue]D"

// FOO_M-LABEL: define hidden i32 @"+[Foo classMethod:]D"
+ (int)classMethod:(int)x {
  return x * 3;
}

// FOO_M-LABEL: define dso_local i32 @"-[Foo exportedInstanceMethod:]D"
- (int)exportedInstanceMethod:(int)x {
  // FOO_M: call i32 @"-[Foo privateValue]D"
  // FOO_M: call i32 @"-[Foo exportedValue]D"
  return x + [self privateValue] + [self exportedValue];
}

// FOO_M-LABEL: define dso_local i32 @"+[Foo exportedClassMethod:]D"
+ (int)exportedClassMethod:(int)x {
  return x * 5;
}

// Hidden property getter and setter (default visibility)
// FOO_M-LABEL: define hidden void @"-[Foo setPrivateValue:]D"

// Exported property getter and setter (explicit default visibility)
// FOO_M-LABEL: define dso_local void @"-[Foo setExportedValue:]D"

@end

//--- main.m
// Consumer of libFoo (separate linkage unit)
#import "foo.h"

int printf(const char *, ...);

int main() {
    Foo *obj = [[Foo alloc] initWithprivateValue:10 exportedValue:20];
    printf("Allocated Foo\n");

    // MAIN_M: call void @"-[Foo setExportedValue:]D_thunk"
    [obj setExportedValue:30];

    // MAIN_M: call i32 @"-[Foo exportedValue]D_thunk"
    printf("Reset exportedValue: %d\n", [obj exportedValue]);

    // MAIN_M: call i32 @"-[Foo exportedInstanceMethod:]D_thunk"
    printf("Exported instance: %d\n", [obj exportedInstanceMethod:10]);

    // MAIN_M: call i32 @"+[Foo exportedClassMethod:]D_thunk"
    printf("Exported class: %d\n", [Foo exportedClassMethod:10]);

    return 0;
}

// Thunks are generated during compilation
// MAIN_M-LABEL: define linkonce_odr hidden void @"-[Foo setExportedValue:]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo exportedValue]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo exportedInstanceMethod:]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"+[Foo exportedClassMethod:]D_thunk"
