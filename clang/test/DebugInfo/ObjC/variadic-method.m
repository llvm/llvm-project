// RUN: %clang_cc1 -o - -emit-llvm -debug-info-kind=limited %s | FileCheck %s

// This test verifies that variadic ObjC methods get the
// DW_TAG_unspecified_parameter marker.

@interface Foo
- (void) Bar: (int) n, ...;
@end

@implementation Foo
- (void) Bar: (int) n, ...
{
  // CHECK: !DISubroutineType(types: ![[NUM:[0-9]+]])
  // CHECK: ![[NUM]] = {{!{null, ![^,]*, ![^,]*, ![^,]*, null}}}
}
@end
