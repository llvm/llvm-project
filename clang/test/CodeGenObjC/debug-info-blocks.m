// RUN: %clang_cc1 -emit-llvm -fblocks -debug-info-kind=limited  -triple x86_64-apple-darwin10 -fobjc-dispatch-method=mixed -x objective-c < %s -o - | FileCheck %s

// rdar://problem/9279956
// Test that we generate the proper debug location for a captured self.
// The second half of this test is in llvm/tests/DebugInfo/debug-info-blocks.ll

// CHECK: define {{.*}}_block_invoke
// CHECK: %[[BLOCK:.*]] = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg
// CHECK-NEXT: store <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %[[BLOCK]], <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %[[ALLOCA:.*]], align
// CHECK-NEXT: call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %[[ALLOCA]], metadata ![[SELF:[0-9]+]], metadata !{{.*}})
// CHECK-NEXT: call void @llvm.dbg.declare(metadata %1** %d, metadata ![[D:[0-9]+]], metadata !{{.*}})

// Test that we do emit scope info for the helper functions, and that the
// parameters to these functions are marked as artificial (so the debugger
// doesn't accidentally step into the function).
// CHECK: define {{.*}} @__copy_helper_block_{{.*}}(i8*, i8*)
// CHECK-NOT: ret
// CHECK: call {{.*}}, !dbg ![[DBG_LINE:[0-9]+]]
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[COPY_LINE:[0-9]+]]
// CHECK: ret void, !dbg ![[COPY_LINE]]
// CHECK: define {{.*}} @__destroy_helper_block_{{.*}}(i8*)
// CHECK-NOT: ret
// CHECK: load {{.*}}, !dbg ![[DESTROY_LINE:[0-9]+]]
// CHECK: ret void, !dbg ![[DESTROY_LINE]]

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
      // CHECK-DAG: [[DBG_LINE]] = !DILocation(line: 0, scope: ![[COPY_SP:[0-9]+]])
      // CHECK-DAG: [[COPY_LINE]] = !DILocation(line: [[@LINE+7]], scope: ![[COPY_SP:[0-9]+]])
      // CHECK-DAG: [[COPY_SP]] = distinct !DISubprogram(name: "__copy_helper_block_8_32o"
      // CHECK-DAG: [[DESTROY_LINE]] = !DILocation(line: [[@LINE+5]], scope: ![[DESTROY_SP:[0-9]+]])
      // CHECK-DAG: [[DESTROY_SP]] = distinct !DISubprogram(name: "__destroy_helper_block_8_32o"
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

int main()
{
	A *a = [[A alloc] init];
	return 0;
}
