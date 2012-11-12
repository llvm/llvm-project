// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -g -S -emit-llvm %s -o - | FileCheck %s
// CHECK: metadata !{i32 {{.*}}, metadata {{.*}}, metadata !"Circle", metadata {{.*}}, i32 11, i64 64, i64 64, i32 0, i32 512, null, metadata {{.*}}, i32 16, i32 0, i32 0} ; [ DW_TAG_structure_type ]
@interface NSObject {
  struct objc_object *isa;
}
@end

@interface Shape : NSObject

@end
@interface Circle : Shape

@end
@implementation Circle

@end
