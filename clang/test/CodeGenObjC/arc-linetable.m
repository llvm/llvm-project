// RUN: %clang_cc1 -emit-llvm -fblocks -fobjc-arc -g -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// Legend: EXP = Return expression, RET = ret instruction

// CHECK: define {{.*}}testNoSideEffect
// CHECK: call void @objc_storeStrong{{.*}}
// CHECK: call void @objc_storeStrong{{.*}} !dbg ![[ARC1:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET1:[0-9]+]]

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
// CHECK: call void @objc_storeStrong{{.*}}
// CHECK: call void @objc_storeStrong{{.*}} !dbg ![[ARC5:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET5:[0-9]+]]

// CHECK: define {{.*}}testVoidNoReturn
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG6:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET6:[0-9]+]]

// CHECK: define {{.*}}testNoCleanupSideEffect
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG7:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET7:[0-9]+]]

// CHECK: define {{.*}}testCleanupVoid
// CHECK: icmp ne {{.*}}!dbg ![[SKIP1:[0-9]+]]
// CHECK: store i32 0, i32* {{.*}}, !dbg ![[RET8:[0-9]+]]
// CHECK: @objc_storeStrong{{.*}}, !dbg ![[ARC8:[0-9]+]]
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

// CHECK: ![[TESTNOSIDEEFFECT:.*]] = {{.*}}[ DW_TAG_subprogram ] [line [[@LINE+1]]] [local] [def] [-[AppDelegate testNoSideEffect:]]
- (int)testNoSideEffect:(NSString *)foo {
  int x = 1;
  // CHECK: ![[ARC1]] = !MDLocation(line: [[@LINE+1]], scope: ![[TESTNOSIDEEFFECT]])
  return 1; // Return expression
  // CHECK: ![[RET1]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
}           // Cleanup + Ret

- (int)testNoCleanup {
  // CHECK: ![[RET2]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  return 1;
}

- (int)testSideEffect:(NSString *)foo {
  // CHECK: ![[MSG3]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  return [self testNoSideEffect :foo];
  // CHECK: ![[RET3]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (int)testMultiline:(NSString *)foo {
  // CHECK: ![[MSG4]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  int r = [self testSideEffect :foo];
  // CHECK: ![[EXP4]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  return r;
  // CHECK: ![[RET4]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (void)testVoid:(NSString *)foo {
  // CHECK: ![[ARC5]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  return;
  // CHECK: ![[RET5]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (void)testVoidNoReturn:(NSString *)foo {
  // CHECK: ![[MSG6]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  [self testVoid :foo];
  // CHECK: ![[RET6]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
}

- (int)testNoCleanupSideEffect {
  // CHECK: ![[MSG7]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  [self testVoid :@"foo"];
  // CHECK: ![[RET7]] = !MDLocation(line: [[@LINE+1]], scope: !{{.*}})
  return 1;
}

- (void)testCleanupVoid:(BOOL)skip withDelegate: (AppDelegate *) delegate {
  static BOOL skip_all;
  // CHECK: ![[SKIP1]] = !MDLocation(line: [[@LINE+1]], scope:
  if (!skip_all) {
    if (!skip) {
      return;
    }
    NSString *s = @"bar";
    if (!skip) {
      [delegate testVoid :s];
    }
  }
  // CHECK: ![[RET8]] = !MDLocation(line: [[@LINE+2]], scope:
  // CHECK: ![[ARC8]] = !MDLocation(line: [[@LINE+1]], scope:
}


@end


int main(int argc, const char** argv) {
  AppDelegate *o = [[AppDelegate alloc] init];
  return [o testMultiline :@"foo"];
}
