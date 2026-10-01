
// RUN: %clang_cc1 -O0 -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O0
// RUN: %clang_cc1 -O1 -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O1
// RUN: %clang_cc1 -O2 -triple spirv64-amd-amdhsa -disable-llvm-passes -emit-llvm-bc -o /dev/null -fdebug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=O2

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

// O1-NOT: Running pass: LowerExpectIntrinsic
// O1-NOT: Running pass: SimplifyCFG
// O1-NOT: Running pass: SROA
// O1-NOT: Running pass: EarlyCSE
// O1-NOT: Running pass: ForceFunctionAttrs
// O1-NOT: Running pass: InferFunctionAttrs
// O1-NOT: Running pass: IPSCCP
// O1-NOT: Running pass: GlobalOpt
// O1-NOT: Running pass: Promote
// O1-NOT: Running pass: DeadArgumentElimination
// O1-NOT: Running pass: InstCombine
// O1-NOT: Running pass: PostOrderFunctionAttrs

// O2-DAG: Running pass: LowerExpectIntrinsic
// O2-DAG: Running pass: SimplifyCFG
// O2-DAG: Running pass: SROA
// O2-DAG: Running pass: EarlyCSE
// O2-DAG: Running pass: ForceFunctionAttrs
// O2-DAG: Running pass: InferFunctionAttrs
// O2-DAG: Running pass: IPSCCP
// O2-DAG: Running pass: GlobalOpt
// O2-DAG: Running pass: Promote
// O2-DAG: Running pass: DeadArgumentElimination
// O2-DAG: Running pass: InstCombine
// O2-DAG: Running pass: PostOrderFunctionAttrs

void foo() {}
