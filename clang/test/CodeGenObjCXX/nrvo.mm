// RUN: %clang_cc1 -emit-llvm -o - -fblocks %s -O1 -fno-inline-functions -triple x86_64-apple-darwin10.0.0 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s

// PR10835
struct X {
  X();
  X(const X&);
  ~X();
};

@interface NRVO
@end

@implementation NRVO
// CHECK: define internal void @"\01-[NRVO getNRVO]"
- (X)getNRVO { 
  X x;
  // CHECK: call void @_ZN1XC1Ev
  // CHECK-NEXT: ret void
  return x;
}
@end

X blocksNRVO() {
  return ^{
    // With the optimizer enabled, the DeadArgElim pass is able to
    // mark the block litteral address argument as unused and later the
    // related block_litteral global variable is removed.
    // This allows to promote this call to a fastcc call.
    // CHECK-LABEL: define internal fastcc void @___Z10blocksNRVOv_block_invoke
    X x;
    // CHECK: call void @_ZN1XC1Ev
    // CHECK-NEXT: ret void
    return x;
  }() ;
}

