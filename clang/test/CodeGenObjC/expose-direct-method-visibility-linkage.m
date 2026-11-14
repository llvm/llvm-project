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

// Hand-written accessors take the property's visibility, like synthesized ones
@property (nonatomic, direct, readonly) int computed __attribute__((visibility("default")));
@property (nonatomic, direct, readonly) int computedHidden;
@property (nonatomic, direct) int handWrittenSetter __attribute__((visibility("default")));
// Class properties are never synthesized
@property (class, nonatomic, direct, readonly) int classValue __attribute__((visibility("default")));

// An attribute on the accessor wins over the property's
@property (nonatomic, direct, readonly) int getterDeclared;
- (int)getterDeclared __attribute__((objc_direct, visibility("default")));
@property (nonatomic, direct, readonly) int definitionExported;
@property (nonatomic, direct, readonly) int definitionHidden __attribute__((visibility("default")));

// Exported readonly property, redeclared readwrite in foo.m
@property (nonatomic, direct, readonly) int counter __attribute__((visibility("default")));
@end

@interface Foo (Cat)
// Category properties are never synthesized
@property (nonatomic, direct, readonly) int categoryValue __attribute__((visibility("default")));
@end

//--- foo.m

// libFoo does not have thunks because the true implementation is not used internally.
// FOO_M-NOT: @{{.*}}_thunk
#import "foo.h"

// The setter takes the visibility of this redeclaration, which has no
// attribute, not that of the exported readonly property in foo.h.
@interface Foo ()
@property (nonatomic, direct, readwrite) int counter;
@end

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

// Hand-written accessors take the property's visibility
// FOO_M-LABEL: define dso_local i32 @"-[Foo computed]D"
- (int)computed {
  return 1;
}

// FOO_M-LABEL: define hidden i32 @"-[Foo computedHidden]D"
- (int)computedHidden {
  return 2;
}

// FOO_M-LABEL: define dso_local void @"-[Foo setHandWrittenSetter:]D"
- (void)setHandWrittenSetter:(int)x {
  _handWrittenSetter = x;
}

// FOO_M-LABEL: define dso_local i32 @"+[Foo classValue]D"
+ (int)classValue {
  return 3;
}

// An attribute on the definition wins over the property's
// FOO_M-LABEL: define dso_local i32 @"-[Foo definitionExported]D"
- (int)definitionExported __attribute__((visibility("default"))) {
  return 4;
}

// FOO_M-LABEL: define hidden i32 @"-[Foo definitionHidden]D"
- (int)definitionHidden __attribute__((visibility("hidden"))) {
  return 5;
}

// Hidden property getter and setter (default visibility)
// FOO_M-LABEL: define hidden void @"-[Foo setPrivateValue:]D"

// Exported property getter and setter (explicit default visibility)
// FOO_M-LABEL: define dso_local void @"-[Foo setExportedValue:]D"

// Synthesized getter of a property with a hand-written setter
// FOO_M-LABEL: define dso_local i32 @"-[Foo handWrittenSetter]D"

// Synthesized getter whose declaration has its own attribute
// FOO_M-LABEL: define dso_local i32 @"-[Foo getterDeclared]D"

// The private setter is hidden, although the readonly property is exported
// FOO_M-LABEL: define dso_local i32 @"-[Foo counter]D"
// FOO_M-LABEL: define hidden void @"-[Foo setCounter:]D"

@end

@implementation Foo (Cat)

// FOO_M-LABEL: define dso_local i32 @"-[Foo categoryValue]D"
- (int)categoryValue {
  return 6;
}

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

    // MAIN_M: call i32 @"-[Foo computed]D_thunk"
    printf("Computed: %d\n", [obj computed]);

    // MAIN_M: call void @"-[Foo setHandWrittenSetter:]D_thunk"
    [obj setHandWrittenSetter:40];

    // MAIN_M: call i32 @"+[Foo classValue]D_thunk"
    printf("Class value: %d\n", [Foo classValue]);

    // MAIN_M: call i32 @"-[Foo categoryValue]D_thunk"
    printf("Category value: %d\n", [obj categoryValue]);

    return 0;
}

// Thunks are generated during compilation
// MAIN_M-LABEL: define linkonce_odr hidden void @"-[Foo setExportedValue:]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo exportedValue]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo exportedInstanceMethod:]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"+[Foo exportedClassMethod:]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo computed]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden void @"-[Foo setHandWrittenSetter:]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"+[Foo classValue]D_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo categoryValue]D_thunk"
