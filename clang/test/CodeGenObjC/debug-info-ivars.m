// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -g %s -o - | FileCheck %s

__attribute((objc_root_class)) @interface NSObject {
	id isa;
}
@end

@interface BaseClass : NSObject
{
    int i;
    unsigned flag_1 : 9;
    unsigned flag_2 : 9;
		unsigned : 1;
		unsigned flag_3 : 9;
}
@end

@implementation BaseClass
@end

// CHECK: metadata !{i32 786445, metadata !{{[0-9]*}}, metadata !"i", metadata !{{[0-9]*}}, i32 10, i64 32, i64 32, i64 0, i32 2, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [i] [line 10, size 32, align 32, offset 0] [protected] [from int]
// CHECK: metadata !{i32 786445, metadata !{{[0-9]*}}, metadata !"flag_1", metadata !{{[0-9]*}}, i32 11, i64 9, i64 32, i64 0, i32 2, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [flag_1] [line 11, size 9, align 32, offset 0] [protected] [from unsigned int]
// CHECK: metadata !{i32 786445, metadata !{{[0-9]*}}, metadata !"flag_2", metadata !{{[0-9]*}}, i32 12, i64 9, i64 32, i64 1, i32 2, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [flag_2] [line 12, size 9, align 32, offset 1] [protected] [from unsigned int]
// CHECK: metadata !{i32 786445, metadata !{{[0-9]*}}, metadata !"flag_3", metadata !{{[0-9]*}}, i32 14, i64 9, i64 32, i64 3, i32 2, metadata !{{[0-9]*}}, null} ; [ DW_TAG_member ] [flag_3] [line 14, size 9, align 32, offset 3] [protected] [from unsigned int]