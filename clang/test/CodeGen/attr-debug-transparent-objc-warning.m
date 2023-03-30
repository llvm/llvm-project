// Pipe stderr to FileCheck since we're checking for a warning
// RUN: %clang -gcodeview -g -emit-llvm -S %s -o - 2>&1 | FileCheck %s

@interface ObjCClass
- (void)foo __attribute__((debug_transparent));
@end

@implementation ObjCClass
- (void)foo {}
@end

// CHECK: warning: 'debug_transparent' attribute is ignored since it is only supported by DWARF
