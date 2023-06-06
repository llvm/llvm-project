// RUN: %clang_cc1 -emit-llvm -fblocks -debug-info-kind=limited  -triple x86_64-apple-darwin10 -fobjc-dispatch-method=mixed -x objective-c < %s -o - | FileCheck %s

// rdar://problem/9279956
// Test that we generate the proper debug location for a captured self.
// The second half of this test is in llvm/tests/DebugInfo/debug-info-blocks.ll

// CHECK: define {{.*}}_block_invoke
// CHECK: store ptr %.block_descriptor, ptr %[[ALLOCA:block.addr]], align
// CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %[[ALLOCA]], metadata ![[SELF:[0-9]+]], metadata !{{.*}})
// CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %d, metadata ![[D:[0-9]+]], metadata !{{.*}})

// Test that we do emit scope info for the helper functions, and that the
// parameters to these functions are marked as artificial (so the debugger
// doesn't accidentally step into the function).
// CHECK: define {{.*}} @__copy_helper_block_{{.*}}(ptr noundef %0, ptr noundef %1)
// CHECK-NOT: ret
// CHECK: call {{.*}}, !dbg ![[DBG_LINE:[0-9]+]]
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[DBG_LINE]]
// CHECK: ret {{.*}}, !dbg ![[DBG_LINE]]
// CHECK: define {{.*}} @__destroy_helper_block_{{.*}}(ptr
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[DESTROY_LINE:[0-9]+]]
// CHECK: ret {{.*}}, !dbg ![[DESTROY_LINE]]

// CHECK-DAG: [[DBG_LINE]] = !DILocation(line: 0, scope: ![[COPY_SP:[0-9]+]])
// CHECK-DAG: [[COPY_SP]] = distinct !DISubprogram(linkageName: "__copy_helper_block_
// CHECK-DAG: [[DESTROY_LINE]] = !DILocation(line: 0, scope: ![[DESTROY_SP:[0-9]+]])
// CHECK-DAG: [[DESTROY_SP]] = distinct !DISubprogram(linkageName: "__destroy_helper_block_
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
      // CHECK-DAG: !DILocalVariable(arg: 1, scope: ![[COPY_SP]], {{.*}}, flags: DIFlagArtificial)
      // CHECK-DAG: !DILocalVariable(arg: 2, scope: ![[COPY_SP]], {{.*}}, flags: DIFlagArtificial)
      // CHECK-DAG: !DILocalVariable(arg: 1, scope: ![[DESTROY_SP]], {{.*}}, flags: DIFlagArtificial)
      run(^{
          // CHECK-DAG: ![[SELF]] = !DILocalVariable(name: "self", scope:{{.*}}, line: [[@LINE+4]],
          // CHECK-DAG: ![[D]] = !DILocalVariable(name: "d", scope:{{.*}}, line: [[@LINE+1]],
          NSMutableDictionary *d = [[NSMutableDictionary alloc] init]; 
          ivar = 42 + (int)[d count];
        });
    }
    return self;
}

@end

int main(void)
{
	A *a = [[A alloc] init];
	return 0;
}
