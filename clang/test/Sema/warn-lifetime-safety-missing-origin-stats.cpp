// RUN: %clang_cc1 -print-stats -Wlifetime-safety %s 2>&1 | FileCheck %s


// CHECK: *** LifetimeSafety Missing Origin per QualType: (QualType : count) :
// CHECK: *** LifetimeSafety Missing Origin per StmtClassName: (StmtClassName : count) :
