
// RUN: %clang_cc1 -O0 -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O0
// RUN: %clang_cc1 -O1 -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O0
// RUN: %clang_cc1 -Og -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O0
// RUN: %clang_cc1 -O2 -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O
// RUN: %clang_cc1 -Os -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O
// RUN: %clang_cc1 -Oz -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O

// O0-NOT: Running pass: LowerExpectIntrinsic
// O0-NOT: Running pass: SimplifyCFG
// O0-NOT: Running pass: SROA
// O0-NOT: Running pass: EarlyCSE
// O0-NOT: Running pass: ForceFunctionAttrs
// O0-NOT: Running pass: InferFunctionAttrs
// O0-NOT: Running pass: IPSCCP
// O0-NOT: Running pass: GlobalOpt
// O0-NOT: Running pass: Promote
// O0-NOT: Running pass: DeadArgumentElimination
// O0-NOT: Running pass: InstCombine
// O0-NOT: Running pass: PostOrderFunctionAttrs

// O-DAG: Running pass: LowerExpectIntrinsic
// O-DAG: Running pass: SimplifyCFG
// O-DAG: Running pass: SROA
// O-DAG: Running pass: EarlyCSE
// O-DAG: Running pass: ForceFunctionAttrs
// O-DAG: Running pass: InferFunctionAttrs
// O-DAG: Running pass: IPSCCP
// O-DAG: Running pass: GlobalOpt
// O-DAG: Running pass: Promote
// O-DAG: Running pass: DeadArgumentElimination
// O-DAG: Running pass: InstCombine
// O-DAG: Running pass: PostOrderFunctionAttrs

void foo() {}
