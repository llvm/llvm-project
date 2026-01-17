// REQUIRES: system-darwin
//
// This test validates that compile units with and without
// -fobjc-direct-precondition-thunk cannot be linked together due to
// symbol name mismatches.
//
// When compiled WITH the flag:
//   - Foo.o defines symbols like "-[Foo instanceMethod]" (exposed)
//   - main.o references "-[Foo instanceMethod]" (exposed)
//
// When compiled WITHOUT the flag:
//   - Foo.o defines symbols like "\01-[Foo instanceMethod]" (hidden, with \01 prefix)
//   - main.o references "\01-[Foo instanceMethod]" (hidden, with \01 prefix)
//
// Mixing flags causes undefined symbol errors at link time because the
// symbol names don't match.

// RUN: rm -rf %t && mkdir -p %t

//--- Foo.h
#import <Foundation/Foundation.h>

@interface Foo : NSObject
- (int)instanceMethod __attribute__((objc_direct)) __attribute__((visibility("default")));
+ (int)classMethod __attribute__((objc_direct)) __attribute__((visibility("default")));
@end

//--- Foo.m
#import "Foo.h"

@implementation Foo
- (int)instanceMethod {
    return 42;
}
+ (int)classMethod {
    return 42;
}
@end

//--- main.m
#import "Foo.h"

int main(int argc, char** argv) {
    Foo *foo = [[Foo alloc] init];
    int a = [foo instanceMethod];
    int b = [Foo classMethod];
    return a + b - 84;
}

// ============================================================================
// Split the file into individual source files
// ============================================================================
// RUN: split-file %s %t

// ============================================================================
// Build all object files
// ============================================================================

// Foo.o WITHOUT flag (defines hidden symbols with \01 prefix)
// RUN: %clang                                     \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   -c %t/Foo.m -I%t -o %t/Foo_without_flag.o

// Foo.o WITH flag (defines exposed symbols without prefix)
// RUN: %clang -fobjc-direct-precondition-thunk    \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   -c %t/Foo.m -I%t -o %t/Foo_with_flag.o

// main.o WITHOUT flag (references hidden symbols with \01 prefix)
// RUN: %clang                                     \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   -c %t/main.m -I%t -o %t/main_without_flag.o

// main.o WITH flag (references exposed symbols without prefix)
// RUN: %clang -fobjc-direct-precondition-thunk    \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   -c %t/main.m -I%t -o %t/main_with_flag.o

// ============================================================================
// Build libraries
// ============================================================================

// Static libraries
// RUN: llvm-ar rcs %t/libFoo_without_flag.a %t/Foo_without_flag.o
// RUN: llvm-ar rcs %t/libFoo_with_flag.a %t/Foo_with_flag.o

// Dynamic libraries
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   -dynamiclib %t/Foo_without_flag.o -o %t/libFoo_without_flag.dylib \
// RUN:   -framework Foundation
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   -dynamiclib %t/Foo_with_flag.o -o %t/libFoo_with_flag.dylib \
// RUN:   -framework Foundation

// ============================================================================
// TEST 1: Static library mismatch - main WITHOUT flag + libFoo WITH flag
// main.o references "\01-[Foo instanceMethod]" but libFoo defines "-[Foo instanceMethod]"
// Linking should FAIL with undefined symbols
// ============================================================================
// RUN: not                                                \
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc    \
// RUN:   %t/main_without_flag.o %t/libFoo_with_flag.a     \
// RUN:   -framework Foundation -o %t/test1 2>&1 | FileCheck %s --check-prefix=LINK-FAIL-1

// LINK-FAIL-1: Undefined symbols

// ============================================================================
// TEST 2: Static library mismatch - main WITH flag + libFoo WITHOUT flag
// main.o references "-[Foo instanceMethod]" but libFoo defines "\01-[Foo instanceMethod]"
// Linking should FAIL with undefined symbols
// ============================================================================
// RUN: not                                                \
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc    \
// RUN:   %t/main_with_flag.o %t/libFoo_without_flag.a     \
// RUN:   -framework Foundation -o %t/test2 2>&1 | FileCheck %s --check-prefix=LINK-FAIL-2

// LINK-FAIL-2: Undefined symbols

// ============================================================================
// TEST 3: Dynamic library mismatch - main WITHOUT flag + libFoo.dylib WITH flag
// Linking should FAIL with undefined symbols
// ============================================================================
// RUN: not                                                \
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc    \
// RUN:   %t/main_without_flag.o %t/libFoo_with_flag.dylib \
// RUN:   -framework Foundation -o %t/test3 2>&1 | FileCheck %s --check-prefix=LINK-FAIL-3

// LINK-FAIL-3: Undefined symbols

// ============================================================================
// TEST 4: Dynamic library mismatch - main WITH flag + libFoo.dylib WITHOUT flag
// Linking should FAIL with undefined symbols
// ============================================================================
// RUN: not                                                \
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc    \
// RUN:   %t/main_with_flag.o %t/libFoo_without_flag.dylib \
// RUN:   -framework Foundation -o %t/test4 2>&1 | FileCheck %s --check-prefix=LINK-FAIL-4

// LINK-FAIL-4: Undefined symbols

// ============================================================================
// TEST 5: Matching flags - both WITHOUT flag (static lib)
// Linking should SUCCEED
// ============================================================================
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc     \
// RUN:   %t/main_without_flag.o %t/libFoo_without_flag.a   \
// RUN:   -framework Foundation -o %t/test5

// ============================================================================
// TEST 6: Matching flags - both WITH flag (static lib)
// Linking should SUCCEED
// ============================================================================
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc     \
// RUN:   %t/main_with_flag.o %t/libFoo_with_flag.a         \
// RUN:   -framework Foundation -o %t/test6

// ============================================================================
// TEST 7: Matching flags - both WITHOUT flag (dynamic lib)
// Linking should FAIL with undefined symbols
// ============================================================================
// RUN: not                                                   \
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc       \
// RUN:   %t/main_without_flag.o %t/libFoo_without_flag.dylib \
// RUN:   -framework Foundation -o %t/test7 2>&1 | FileCheck %s --check-prefix=LINK-FAIL-7

// LINK-FAIL-7: Undefined symbols

// ============================================================================
// TEST 8: Matching flags - both WITH flag (dynamic lib)
// Linking should SUCCEED: the symbols are exposed through default visibility
// ============================================================================
// RUN: %clang -target arm64-apple-macos11.0 -fobjc-arc \
// RUN:   %t/main_with_flag.o %t/libFoo_with_flag.dylib \
// RUN:   -framework Foundation -o %t/test8
