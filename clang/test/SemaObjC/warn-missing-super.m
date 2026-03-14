@protocol NSCopying @end

__attribute__((objc_root_class))
@interface NSObject <NSCopying>
- (void)dealloc;
@end

@implementation NSObject
- (void)dealloc {
  // Root class, shouldn't warn
}
- (void)finalize {
  // Root class, shouldn't warn
}
@end

@interface Subclass1 : NSObject
- (void)dealloc;
- (void)finalize;
@end

@implementation Subclass1
- (void)dealloc {
}
- (void)finalize {
}
@end

@interface Subclass2 : NSObject
- (void)dealloc;
- (void)finalize;
@end

@implementation Subclass2
- (void)dealloc {
  [super dealloc];  // Shouldn't warn
}
- (void)finalize {
  [super finalize];  // Shouldn't warn
}
@end

// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: warn-missing-super.m:24:1: warning: method possibly missing a [super dealloc] call
// CHECK: 1 warning generated.

// RUN: %clang_cc1 -fsyntax-only -fobjc-gc %s 2>&1 | FileCheck --check-prefix=CHECK-GC %s
// CHECK-GC: warn-missing-super.m:24:1: warning: method possibly missing a [super dealloc] call
// CHECK-GC: warn-missing-super.m:26:1: warning: method possibly missing a [super finalize] call
// CHECK-GC: 2 warnings generated.

// RUN: %clang_cc1 -fsyntax-only -fobjc-gc-only %s 2>&1 | FileCheck --check-prefix=CHECK-GC-ONLY %s
// CHECK-GC-ONLY: warn-missing-super.m:26:1: warning: method possibly missing a [super finalize] call
// CHECK-GC-ONLY: 1 warning generated.

// RUN: not %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin10 -fobjc-arc %s 2>&1 | FileCheck --check-prefix=CHECK-ARC %s
// CHECK-ARC: warn-missing-super.m:36:10: error: ARC forbids explicit message send of 'dealloc'
// CHECK-ARC: 1 error generated.
