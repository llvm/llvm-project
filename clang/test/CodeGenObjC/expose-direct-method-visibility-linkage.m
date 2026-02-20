// REQUIRES: system-darwin

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test 1: Check IR from library implementation (visibility attributes)
// RUN: %clang -target arm64-apple-darwin -fobjc-arc \
// RUN:   -fobjc-direct-precondition-thunk -S -emit-llvm -o - %t/foo.m \
// RUN:   -I %t | FileCheck %s --check-prefix=FOO_M

// Test 2: Check IR from main (consumer)
// RUN: %clang -target arm64-apple-darwin -fobjc-arc \
// RUN:   -fobjc-direct-precondition-thunk -S -emit-llvm -o - %t/main.m \
// RUN:   -I %t | FileCheck %s --check-prefix=MAIN_M

// Test 3: Build libFoo.dylib from foo.m
// RUN: %clang -fobjc-direct-precondition-thunk -target arm64-apple-darwin \
// RUN:   -fobjc-arc -dynamiclib %t/foo.m -I %t \
// RUN:   -framework Foundation \
// RUN:   -install_name @rpath/libFoo.dylib \
// RUN:   -o %t/libFoo.dylib

// Test 4: Verify visibility works correctly in dylib
// RUN: llvm-nm -g %t/libFoo.dylib | FileCheck %s --check-prefix=DYLIB

// Hidden visibility methods should NOT be exported
// DYLIB-NOT: -[Foo instanceMethod:]
// DYLIB-NOT: +[Foo classMethod:]
// DYLIB-NOT: -[Foo privateValue:]
// DYLIB-NOT: -[Foo setPrivateValue:]

// Default visibility methods SHOULD be exported
// DYLIB: {{[0-9a-f]+}} T _+[Foo exportedClassMethod:]
// DYLIB: {{[0-9a-f]+}} T _-[Foo exportedInstanceMethod:]
// DYLIB: {{[0-9a-f]+}} T _-[Foo exportedValue]
// DYLIB: {{[0-9a-f]+}} T _-[Foo setExportedValue:]

// Test 5: Compile main.m
// RUN: %clang -fobjc-direct-precondition-thunk -target arm64-apple-darwin \
// RUN:   -fobjc-arc -c %t/main.m -I %t -o %t/main.o

// Test 6: Link main with libFoo.dylib
// RUN: %clang -target arm64-apple-darwin -fobjc-arc \
// RUN:   %t/main.o -L%t -lFoo \
// RUN:   -Wl,-rpath,%t \
// RUN:   -framework Foundation \
// RUN:   -o %t/main

// Test 7: Verify symbols in main executable
// RUN: llvm-nm %t/main | FileCheck %s --check-prefix=EXE

// Thunks should be defined locally
// EXE-DAG: {{[0-9a-f]+}} t _+[Foo exportedClassMethod:]_thunk
// EXE-DAG: {{[0-9a-f]+}} t _-[Foo exportedInstanceMethod:]_thunk
// EXE-DAG: {{[0-9a-f]+}} t _-[Foo exportedValue]_thunk
// EXE-DAG: {{[0-9a-f]+}} t _-[Foo setExportedValue:]_thunk

// Actual methods should be undefined
// EXE-DAG: U _+[Foo exportedClassMethod:]
// EXE-DAG: U _-[Foo exportedInstanceMethod:]
// EXE-DAG: U _-[Foo exportedValue]
// EXE-DAG: U _-[Foo setExportedValue:]

//--- foo.h
// Header for libFoo
#import <Foundation/Foundation.h>

@interface Foo : NSObject
// Direct properties with default hidden visibility
@property (nonatomic, direct) int privateValue;

// Direct properties with explicit default visibility
@property (nonatomic, direct) int exportedValue __attribute__((visibility("default")));

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
  self = [super init];
  if (self) {
    _privateValue = x;
    _exportedValue = y;
  }
  return self;
}

// FOO_M-LABEL: define hidden i32 @"-[Foo instanceMethod:]"
- (int)instanceMethod:(int)x {
  // Compiler is smart enough to know that self is non-nil, so we dispatch to
  // true implementation.
  // FOO_M: call i32 @"-[Foo privateValue]"
  // FOO_M: call i32 @"-[Foo exportedValue]"
  return x + [self privateValue] + [self exportedValue];
}

// Hidden property getter and setter (default visibility)
// FOO_M-LABEL: define hidden i32 @"-[Foo privateValue]"

// Exported property getter and setter (explicit default visibility)
// FOO_M-LABEL: define dso_local i32 @"-[Foo exportedValue]"

// FOO_M-LABEL: define hidden i32 @"+[Foo classMethod:]"
+ (int)classMethod:(int)x {
  return x * 3;
}

// FOO_M-LABEL: define dso_local i32 @"-[Foo exportedInstanceMethod:]"
- (int)exportedInstanceMethod:(int)x {
  // FOO_M: call i32 @"-[Foo privateValue]"
  // FOO_M: call i32 @"-[Foo exportedValue]"
  return x + [self privateValue] + [self exportedValue];
}

// FOO_M-LABEL: define dso_local i32 @"+[Foo exportedClassMethod:]"
+ (int)exportedClassMethod:(int)x {
  return x * 5;
}

// Hidden property getter and setter (default visibility)
// FOO_M-LABEL: define hidden void @"-[Foo setPrivateValue:]"

// Exported property getter and setter (explicit default visibility)
// FOO_M-LABEL: define dso_local void @"-[Foo setExportedValue:]"

@end

//--- main.m
// Consumer of libFoo (separate linkage unit)
#import "foo.h"
#include <stdio.h>

int main() {
@autoreleasepool {
    Foo *obj = [[Foo alloc] initWithprivateValue:10 exportedValue:20];
    printf("Allocated Foo\n");

    // MAIN_M: call void @"-[Foo setExportedValue:]_thunk"
    [obj setExportedValue:30];

    // MAIN_M: call i32 @"-[Foo exportedValue]_thunk"
    printf("Reset exportedValue: %d\n", [obj exportedValue]);

    // MAIN_M: call i32 @"-[Foo exportedInstanceMethod:]_thunk"
    printf("Exported instance: %d\n", [obj exportedInstanceMethod:10]);

    // MAIN_M: call i32 @"+[Foo exportedClassMethod:]_thunk"
    printf("Exported class: %d\n", [Foo exportedClassMethod:10]);
}
    return 0;
}

// Thunks are generated during compilation
// MAIN_M-LABEL: define linkonce_odr hidden void @"-[Foo setExportedValue:]_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo exportedValue]_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"-[Foo exportedInstanceMethod:]_thunk"
// MAIN_M-LABEL: define linkonce_odr hidden i32 @"+[Foo exportedClassMethod:]_thunk"
