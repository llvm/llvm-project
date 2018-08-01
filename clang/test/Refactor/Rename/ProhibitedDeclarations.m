@protocol P1
@end

void dontRenameProtocol() {
  Protocol *p = @protocol(P1);
}
// RUN: not clang-refactor-test rename-initiate -at=%s:5:3 -new-name=foo %s 2>&1 | FileCheck %s
// CHECK: error: could not rename symbol at the given location

#include <objc-system-header.h>

@interface MyClass: MySystemClass

- (void)someMethod:(int)x with:(int)y;

@end

@implementation MyClass

- (void)someMethod:(int)x with:(int)y {
}

@end

// RUN: not clang-refactor-test rename-initiate -at=%s:14:9 -at=%s:20:9 -at=%s:28:9  -new-name=foo:bar %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM %s
// CHECK-SYSTEM: method 'someMethod:with:' cannot be renamed because it overrides a method declared in a system framework

@interface MySubClass: MyClass
@end
@implementation MySubClass
- (void)someMethod:(int)x with:(int)y {
}
@end

// RUN: not clang-refactor-test rename-initiate -at=%s:31:9  -new-name=foo:bar %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM %s
