// RUN: %clang_cc1 -emit-llvm -o - -fblocks %s -O1 -fno-experimental-new-pass-manager -triple x86_64-apple-darwin10.0.0 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s

// PR10835 / <rdar://problem/10050178>
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
  // CHECK: tail call void @_ZN1XC1Ev
  // CHECK-NEXT: ret void
  return x;
}
@end

X blocksNRVO() {
  return ^{
    // CHECK-LABEL: define internal void @___Z10blocksNRVOv_block_invoke
    X x;
    // CHECK: tail call void @_ZN1XC1Ev
    // CHECK-NEXT: ret void
    return x;
  }() ;
}

