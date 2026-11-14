// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// In SYCL device code, annotation strings, arguments, and the
// @llvm.global.annotations table live in the global address space, and
// annotated locals and parameters are addressed through generic pointers.
// Fields are covered by CodeGenSYCL/field-annotate-addr-space.cpp.

// LLVM-DAG: [[PARAM:@\.str[.0-9a-z]*]] = private unnamed_addr addrspace(1) constant [9 x i8] c"on_param\00", section "llvm.metadata"
// LLVM-DAG: [[FILE:@\.str[.0-9a-z]*]] = private unnamed_addr addrspace(1) constant [{{[0-9]+}} x i8] c"{{.*}}annotate-address-space.cpp\00", section "llvm.metadata"
// LLVM-DAG: [[LOCAL:@\.str[.0-9a-z]*]] = private unnamed_addr addrspace(1) constant [9 x i8] c"on_local\00", section "llvm.metadata"
// LLVM-DAG: [[ARGS:@\.args[.0-9a-z]*]] = private unnamed_addr addrspace(1) constant { i32 } { i32 7 }, section "llvm.metadata"
// LLVM-DAG: [[GLOBAL:@\.str[.0-9a-z]*]] = private unnamed_addr addrspace(1) constant [10 x i8] c"on_global\00", section "llvm.metadata"
// LLVM-DAG: [[FUNC:@\.str[.0-9a-z]*]] = private unnamed_addr addrspace(1) constant [8 x i8] c"on_func\00", section "llvm.metadata"

// The annotated function is in the program address space and is cast to the
// table's address space.
// LLVM-DAG: @llvm.global.annotations = appending addrspace(1) global [2 x { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1) }] [{ ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1) } { ptr addrspace(1) @g, ptr addrspace(1) [[GLOBAL]], ptr addrspace(1) [[FILE]], i32 {{[0-9]+}}, ptr addrspace(1) null }, { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1) } { ptr addrspace(1) addrspacecast (ptr @_Z3usei to ptr addrspace(1)), ptr addrspace(1) [[FUNC]], ptr addrspace(1) [[FILE]], i32 {{[0-9]+}}, ptr addrspace(1) null }], section "llvm.metadata"

int g __attribute__((annotate("on_global")));

[[clang::sycl_external]] __attribute__((annotate("on_func"))) int use(int p __attribute__((annotate("on_param")))) {
  int x __attribute__((annotate("on_local", 7)));
  x = p;
  return x + g;
}

// CIR-LABEL: cir.func {{.*}} @_Z3usei
// CIR:   %[[P_ALLOCA:.*]] = cir.alloca "p" align(4) init : !cir.ptr<!s32i>
// CIR:   %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) : !cir.ptr<!s32i>
// CIR:   %[[X:.*]] = cir.cast address_space %[[X_ALLOCA]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:   %[[P:.*]] = cir.cast address_space %[[P_ALLOCA]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:   cir.var_annotation %[[P]] [#cir.annotation<"on_param">] : !cir.ptr<!s32i, target_address_space(4)>
// CIR:   cir.var_annotation %[[X]] [#cir.annotation<"on_local", [7 : i32]>] : !cir.ptr<!s32i, target_address_space(4)>

// LLVM-LABEL: define {{.*}}spir_func {{.*}}i32 @_Z3usei(
// LLVM-DAG: call void @llvm.var.annotation.p4.p1(ptr addrspace(4) %{{.*}}, ptr addrspace(1) [[PARAM]], ptr addrspace(1) [[FILE]], i32 [[@LINE-15]], ptr addrspace(1) null)
// LLVM-DAG: call void @llvm.var.annotation.p4.p1(ptr addrspace(4) %{{.*}}, ptr addrspace(1) [[LOCAL]], ptr addrspace(1) [[FILE]], i32 [[@LINE-15]], ptr addrspace(1) [[ARGS]])
