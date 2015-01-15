// RUN: %clang_cc1 -emit-llvm -fblocks -g  -triple x86_64-apple-darwin10 -fobjc-dispatch-method=mixed -x objective-c < %s -o - | FileCheck %s

// rdar://problem/9279956
// Test that we generate the proper debug location for a captured self.
// The second half of this test is in llvm/tests/DebugInfo/debug-info-blocks.ll

// CHECK: define {{.*}}_block_invoke
// CHECK: %[[BLOCK:.*]] = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg
// CHECK-NEXT: store <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %[[BLOCK]], <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %[[ALLOCA:.*]], align
// CHECK-NEXT: call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %[[ALLOCA]], metadata ![[SELF:[0-9]+]], metadata !{{.*}})
// CHECK-NEXT: call void @llvm.dbg.declare(metadata %1** %d, metadata ![[D:[0-9]+]], metadata !{{.*}})

// rdar://problem/14386148
// Test that we don't emit bogus line numbers for the helper functions.
// Test that we do emit scope info for the helper functions.
// CHECK: define {{.*}} @__copy_helper_block_{{.*}}(i8*, i8*)
// CHECK-NOT: ret
// CHECK: call {{.*}}, !dbg ![[DBG_LINE:[0-9]+]]
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[COPY_LINE:[0-9]+]]
// CHECK: define {{.*}} @__destroy_helper_block_{{.*}}(i8*)
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[DESTROY_LINE:[0-9]+]]

// CHECK-DAG: [[DBG_LINE]] = !MDLocation(line: 0, scope: ![[COPY_SP:[0-9]+]])
// CHECK-DAG: [[COPY_LINE]] = !MDLocation(line: 0, scope: ![[COPY_SP:[0-9]+]])
// CHECK-DAG: [[COPY_SP]] = {{.*}}[ DW_TAG_subprogram ]{{.*}}[__copy_helper_block_]
// CHECK-DAG: [[DESTROY_LINE]] = !MDLocation(line: 0, scope: ![[DESTROY_SP:[0-9]+]])
// CHECK-DAG: [[DESTROY_SP]] = {{.*}}[ DW_TAG_subprogram ]{{.*}}[__destroy_helper_block_]
typedef unsigned int NSUInteger;

@protocol NSObject
@end  
   
@interface NSObject <NSObject>
- (id)init;
+ (id)alloc;
@end 

@interface NSDictionary : NSObject 
- (NSUInteger)count;
@end    

@interface NSMutableDictionary : NSDictionary  
@end       

@interface A : NSObject {
@public
    int ivar;
}
@end

static void run(void (^block)(void))
{
    block();
}

@implementation A

- (id)init
{
    if ((self = [super init])) {
      run(^{
          // CHECK-DAG: ![[SELF]] = {{.*}} [ DW_TAG_auto_variable ] [self] [line [[@LINE+4]]]
          // CHECK-DAG: ![[D]] = {{.*}} [d] [line [[@LINE+1]]]
          NSMutableDictionary *d = [[NSMutableDictionary alloc] init]; 
          ivar = 42 + (int)[d count];
        });
    }
    return self;
}

@end

int main()
{
	A *a = [[A alloc] init];
	return 0;
}
