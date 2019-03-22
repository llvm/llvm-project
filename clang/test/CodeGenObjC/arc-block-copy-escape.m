// RUN: %clang_cc1 -fobjc-arc -fblocks -emit-llvm %s -o - | FileCheck %s

typedef void (^block_t)(void);
void use_block(block_t);
void use_int(int);

// rdar://problem/10211676

void test0(int i) {
  block_t block = ^{ use_int(i); };
  // CHECK-LABEL:   define {{.*}}void @test0(
  // CHECK-NOT: @llvm.objc.retainBlock(
  // CHECK:     ret void
}

void test1(int i) {
  id block = ^{ use_int(i); };
  // CHECK-LABEL:   define {{.*}}void @test1(
  // CHECK:     call {{.*}}i8* @llvm.objc.retainBlock(i8* {{%.*}}) [[NUW:#[0-9]+]]
  // CHECK-NOT: !clang.arc.copy_on_escape
  // CHECK:     ret void
}

// CHECK: attributes [[NUW]] = { nounwind }
