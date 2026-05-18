// RUN: %clang_cc1 -emit-llvm -fblocks -fobjc-arc -debug-info-kind=standalone -dwarf-version=4 -gno-column-info -disable-llvm-passes -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// Legend: EXP = Return expression, RET = ret instruction

// CHECK: define {{.*}}testNoSideEffect
// CHECK: call void @llvm.objc.storeStrong{{.*}}
// CHECK: call void @llvm.objc.storeStrong{{.*}} !dbg ![[RET1:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET1]]

// CHECK: define {{.*}}testNoCleanup
// CHECK: ret {{.*}} !dbg ![[RET2:[0-9]+]]

// CHECK: define {{.*}}testSideEffect
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG3:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET3:[0-9]+]]

// CHECK: define {{.*}}testMultiline
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG4:[0-9]+]]
// CHECK: load{{.*}} !dbg ![[EXP4:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET4:[0-9]+]]

// CHECK: define {{.*}}testVoid
// CHECK: call void @llvm.objc.storeStrong{{.*}}
// CHECK: call void @llvm.objc.storeStrong{{.*}} !dbg ![[RET5:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET5]]

// CHECK: define {{.*}}testVoidNoReturn
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG6:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET6:[0-9]+]]

// CHECK: define {{.*}}testNoCleanupSideEffect
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG7:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET7:[0-9]+]]

// CHECK: define {{.*}}testCleanupVoid
// CHECK: icmp ne {{.*}}!dbg ![[SKIP1:[0-9]+]]
// CHECK: store i32 0, ptr {{.*}}, !dbg ![[RET8:[0-9]+]]
// CHECK: @llvm.objc.storeStrong{{.*}}, !dbg ![[RET8]]
// CHECK: ret {{.*}} !dbg ![[RET8]]

typedef signed char BOOL;

@interface NSObject
+ (id)alloc;
- (id)init;
- (id)retain;
@end

@class NSString;

@interface AppDelegate : NSObject

@end

@implementation AppDelegate : NSObject

// CHECK: ![[TESTNOSIDEEFFECT:.*]] = distinct !DISubprogram(name: "-[AppDelegate testNoSideEffect:]"
// CHECK-SAME:                                              line: [[@LINE+2]]
// CHECK-SAME:                                              DISPFlagLocalToUnit | DISPFlagDefinition
- (int)testNoSideEffect:(NSString *)foo {
  int x = 1;
  return 1; // Return expression
  // CHECK: ![[RET1]] = !DILocation(line: [[@LINE+1]], scope: ![[TESTNOSIDEEFFECT]])
}           // Cleanup + Ret

- (int)testNoCleanup {
  // CHECK: ![[RET2]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  return 1;
}

- (int)testSideEffect:(NSString *)foo {
  // CHECK: ![[MSG3]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  return [self testNoSideEffect :foo];
  // CHECK: ![[RET3]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (int)testMultiline:(NSString *)foo {
  // CHECK: ![[MSG4]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  int r = [self testSideEffect :foo];
  // CHECK: ![[EXP4]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  return r;
  // CHECK: ![[RET4]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (void)testVoid:(NSString *)foo {
  return;
  // CHECK: ![[RET5]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (void)testVoidNoReturn:(NSString *)foo {
  // CHECK: ![[MSG6]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  [self testVoid :foo];
  // CHECK: ![[RET6]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (int)testNoCleanupSideEffect {
  // CHECK: ![[MSG7]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  [self testVoid :@"foo"];
  // CHECK: ![[RET7]] = !DILocation(line: [[@LINE+1]], scope: !{{.*}})
  return 1;
}

- (void)testCleanupVoid:(BOOL)skip withDelegate: (AppDelegate *) delegate {
  static BOOL skip_all;
  // CHECK: ![[SKIP1]] = !DILocation(line: [[@LINE+1]], scope:
  if (!skip_all) {
    if (!skip) {
      return;
    }
    NSString *s = @"bar";
    if (!skip) {
      [delegate testVoid :s];
    }
  }
  // CHECK: ![[RET8]] = !DILocation(line: [[@LINE+1]], scope:
}


@end


int main(int argc, const char** argv) {
  AppDelegate *o = [[AppDelegate alloc] init];
  return [o testMultiline :@"foo"];
}
