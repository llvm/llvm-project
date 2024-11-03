// RUN: %clang_cc1 -emit-llvm -triple i386-apple-macosx10.7.2 %s -o - | FileCheck %s 

@interface CNumber
- (double _Complex)sum;
@end

double _Complex foo(CNumber *x) {
  return [x sum];
}

// CHECK:      [[R:%.*]] = phi double [ [[R1:%.*]], [[MSGCALL:%.*]] ], [ 0.000000e+00, [[NULLINIT:%.*]] ]
// CHECK-NEXT: [[I:%.*]] = phi double [ [[I1:%.*]], [[MSGCALL]] ], [ 0.000000e+00, [[NULLINIT]] ]
// CHECK: store double [[R]]
// CHECK: store double [[I]]
