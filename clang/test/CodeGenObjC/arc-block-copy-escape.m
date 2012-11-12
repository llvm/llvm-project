// RUN: %clang_cc1 -fobjc-arc -fblocks -emit-llvm %s -o - | FileCheck %s

typedef void (^block_t)(void);
void use_block(block_t);
void use_int(int);

// rdar://problem/10211676

void test0(int i) {
  block_t block = ^{ use_int(i); };
  // CHECK:   define void @test0(
  // CHECK:     call i8* @objc_retainBlock(i8* {{%.*}}) nounwind, !clang.arc.copy_on_escape
  // CHECK:     ret void
}

void test1(int i) {
  id block = ^{ use_int(i); };
  // CHECK:   define void @test1(
  // CHECK:     call i8* @objc_retainBlock(i8* {{%.*}}) nounwind
  // CHECK-NOT: !clang.arc.copy_on_escape
  // CHECK:     ret void
}
