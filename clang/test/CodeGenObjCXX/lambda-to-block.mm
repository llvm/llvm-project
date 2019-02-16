// RUN: %clang_cc1 -x objective-c++ -fblocks -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -std=c++1z -emit-llvm -o - %s | FileCheck %s

// rdar://31385153
// Shouldn't crash!

void takesBlock(void (^)(void));

struct Copyable {
  Copyable(const Copyable &x);
};

void hasLambda(Copyable x) {
  takesBlock([x] () { });
}
// CHECK-LABEL: define internal void @__copy_helper_block_
// CHECK: call void @"_ZZ9hasLambda8CopyableEN3$_0C1ERKS0_"
// CHECK-LABEL: define internal void @"_ZZ9hasLambda8CopyableEN3$_0C2ERKS0_"
// CHECK: call void @_ZN8CopyableC1ERKS_
