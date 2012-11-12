// RUN: %clang_cc1 -emit-llvm -triple i386-apple-macosx10.7.2 %s -o - | FileCheck %s 
// rdar://10331109

@interface CNumber
- (double _Complex)sum;
@end

double _Complex foo(CNumber *x) {
  return [x sum];
}

// CHECK: [[T4:%.*]] = phi double [ 0.000000e+00, [[NULLINIT:%.*]] ], [ [[R1:%.*]], [[MSGCALL:%.*]] ]
// CHECK: [[T5:%.*]] = phi double [ 0.000000e+00, [[NULLINIT:%.*]] ], [ [[I1:%.*]], [[MSGCALL:%.*]] ]

// CHECK: store double [[T4]]
// CHECK: store double [[T5]]
