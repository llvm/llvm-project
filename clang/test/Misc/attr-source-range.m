// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s
// Coverage for attributes attached to some ObjCMethodDecl

@interface AllocAlignMethods

- (char *)allocate:(char *)alignment __attribute__((alloc_align(1)));
// CHECK: attr-source-range.m:[[@LINE-1]]:65:{[[@LINE-1]]:21-[[@LINE-1]]:37}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type

- (char *)allocate:(unsigned long)alignment context:(char *)context
    __attribute__((alloc_align(2)));
// CHECK: attr-source-range.m:[[@LINE-1]]:32:{[[@LINE-2]]:54-[[@LINE-2]]:68}: error: 'alloc_align' attribute argument may only refer to a function parameter of integer type
@end
