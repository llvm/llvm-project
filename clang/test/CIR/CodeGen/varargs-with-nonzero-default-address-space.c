// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fcuda-is-device -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// TODO(cir): Add aggregate va_arg once va_arg ABI lowering supports SPIR-V.
int testva(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  int v = __builtin_va_arg(ap2, int);
  __builtin_va_end(ap2);
  __builtin_va_end(ap);
  return v;
}

// CIR-LABEL: cir.func {{.*}} @testva(
// CIR:   cir.va_start %[[AP:.+]] : !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(4)>
// CIR:   cir.va_copy %[[AP]] to %[[AP2:.+]] : !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(4)>, !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(4)>
// CIR:   cir.va_arg %[[AP2]] : (!cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(4)>) -> !s32i
// CIR:   cir.va_end %[[AP2]] : !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(4)>
// CIR:   cir.va_end %[[AP]] : !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(4)>

// LLVM-LABEL: define spir_func i32 @testva(
// LLVM:   call void @llvm.va_start.p4(ptr addrspace(4) %[[AP:.*]])
// LLVM:   call void @llvm.va_copy.p4(ptr addrspace(4) %[[AP2:.*]], ptr addrspace(4) %[[AP]])
// LLVMCIR: va_arg ptr addrspace(4) %[[AP2]], i32
// LLVM:   call void @llvm.va_end.p4(ptr addrspace(4) %[[AP2]])
// LLVM:   call void @llvm.va_end.p4(ptr addrspace(4) %[[AP]])
