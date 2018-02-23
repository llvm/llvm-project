// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -emit-llvm -triple x86_64-apple-darwin -o - %s | FileCheck %s
//
// Test that debug location is generated for a captured "self" inside
// a block.
//
// This test is split into two parts, this one for the frontend, and
// then llvm/test/DebugInfo/debug-info-block-captured-self.ll to
// ensure that DW_AT_location is generated for the captured self.
@class T;
@interface S
@end
@interface Mode
-(int) count;
@end
@interface Context
@end
@interface ViewController
@property (nonatomic, readwrite, strong) Context *context;
@end
typedef enum {
    Unknown = 0,
} State;
@interface Main : ViewController
{
    T * t1;
    T * t2;
}
@property(readwrite, nonatomic) State state;
@end
@implementation Main
- (id) initWithContext:(Context *) context
{
    t1 = [self.context withBlock:^(id obj){
        id *mode1;
	t2 = [mode1 withBlock:^(id object){
	    Mode *mode2 = object;
	    if ([mode2 count] != 0) {
	      self.state = 0;
	    }
	  }];
      }];
}
@end
// The important part of this test is that there is a dbg.value
// intrinsic associated with the implicit .block_descriptor argument
// of the block. We also test that this value gets alloca'd, so the
// register llocator won't accidentally kill it.

// outer block:
// CHECK: define internal void {{.*}}_block_invoke{{.*}}

// inner block:
// CHECK: define internal void {{.*}}_block_invoke{{.*}}
// CHECK:        %[[MEM1:.*]] = alloca i8*, align 8
// CHECK-NEXT:   %[[MEM2:.*]] = alloca i8*, align 8
// CHECK-NEXT:   [[DBGADDR:%.*]] = alloca [[BLOCK_T:<{.*}>]]*, align 8
// CHECK:        store i8* [[BLOCK_DESC:%.*]], i8** %[[MEM1]], align 8
// CHECK:        %[[TMP0:.*]] = load i8*, i8** %[[MEM1]]
// CHECK:        call void @llvm.dbg.value(metadata i8* %[[TMP0]], i64 0, metadata ![[BDMD:[0-9]+]], metadata !{{.*}})
// CHECK:        call void @llvm.dbg.declare(metadata i8* [[BLOCK_DESC]], metadata ![[BDMD:[0-9]+]], metadata !{{.*}})
// CHECK:        store [[BLOCK_T]]* {{%.*}}, [[BLOCK_T]]** [[DBGADDR]], align 8
// CHECK:        call void @llvm.dbg.declare(metadata [[BLOCK_T]]** [[DBGADDR]], metadata ![[SELF:.*]], metadata !{{.*}})
// make sure we are still in the same function
// CHECK: define {{.*}}__copy_helper_block_
// Metadata
// CHECK: ![[MAIN:.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Main"
// CHECK-SAME:                            line: 23,
// CHECK: ![[PMAIN:.*]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[MAIN]],
// CHECK: ![[BDMD]] = !DILocalVariable(name: ".block_descriptor", arg:
// CHECK: ![[SELF]] = !DILocalVariable(name: "self"
// CHECK-NOT:                          arg:
// CHECK-SAME:                         line: 40,
