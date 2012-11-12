// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s
// self and _cmd are marked as DW_AT_artificial. 
// myarg is not marked as DW_AT_artificial.

// CHECK: metadata !{i32 {{.*}}, metadata !9, metadata !"self", metadata !15, i32 16777232, metadata !30, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [self] [line 16]
// CHECK: metadata !{i32 {{.*}}, metadata !9, metadata !"_cmd", metadata !15, i32 33554448, metadata !33, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [_cmd] [line 16]
// CHECK: metadata !{i32 {{.*}}, metadata !9, metadata !"myarg", metadata !6, i32 50331664, metadata !24, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [myarg] [line 16]


@interface MyClass {
}
- (id)init:(int) myarg;
@end

@implementation MyClass
- (id) init:(int) myarg
{
    return self;
}
@end
