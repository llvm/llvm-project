// CUDA device variable registration is covered by CodeGenCUDA/device-stub.cu.
// This test only checks HIP-specific __hipRegisterVar emission.

// RUN: echo -n "GPU binary would be here." > %t

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t \
// RUN:            -emit-llvm %s -o %t.cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

__attribute__((device)) int device_var;
__attribute__((constant)) int constant_var;

// CIR: cir.func private @__hipRegisterVar
// CIR-LABEL: cir.func internal private @__hip_register_globals
// CIR: cir.call @__hipRegisterVar
// CIR: cir.call @__hipRegisterVar
// CIR-DAG: cir.global{{.*}} @device_var = {{.*}}cu.var_registration = #cir.cu.var_registration<device_var, Variable>
// CIR-DAG: cir.global{{.*}} @constant_var = {{.*}}cu.var_registration = #cir.cu.var_registration<constant_var, Variable, constant>

// LLVM-LABEL: define internal void @__hip_register_globals
// LLVM: call void @__hipRegisterVar(ptr %[[HANDLE:[^,]+]], ptr @device_var, ptr @[[DEVICE_VAR_NAME:[^,]+]], ptr @[[DEVICE_VAR_NAME]], i32 0, i64 4, i32 0, i32 0)
// LLVM: call void @__hipRegisterVar(ptr %[[HANDLE]], ptr @constant_var, ptr @[[CONSTANT_VAR_NAME:[^,]+]], ptr @[[CONSTANT_VAR_NAME]], i32 0, i64 4, i32 1, i32 0)
