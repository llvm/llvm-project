// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:   -emit-cir -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.pre.cir
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -fclangir \
// RUN:   -emit-llvm -O0 %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 \
// RUN:   -emit-llvm -O0 %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll

#define NULL ((void *)0)

void test_storage(private char **arg_private, local char **arg_local,
                  global char **arg_global, constant char **arg_constant,
                  generic char **arg_generic) {
  *arg_private = 0;
  *arg_local = 0;
  *arg_global = 0;
  *arg_constant = 0;
  *arg_generic = 0;
}

// CIR-LABEL: cir.func {{.*}}@test_storage
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_private)>
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_local)>
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_global)>
// CIR: cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_constant)>
// CIR-NOT: cir.cast address_space
// CIR: cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR-NOT: cir.cast address_space
// CIR: cir.return

// LLVM-LABEL: define {{.*}}void @test_storage
// LLVM: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr addrspace(4) %{{.*}}
// LLVM: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr addrspace(4) %{{.*}}
// LLVM: store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr addrspace(4) %{{.*}}
// LLVM: store ptr addrspace(2) null, ptr addrspace(4) %{{.*}}
// LLVM: store ptr addrspace(4) null, ptr addrspace(4) %{{.*}}

void test_init(void) {
  private char *p1 = 0;
  local char *p2 = (local char *)0;
  global char *p3 = NULL;
  constant char *p4 = 0;
  generic char *p5 = NULL;
}

// LLVM-LABEL: define {{.*}}void @test_init
// LLVM: store ptr addrspacecast (ptr addrspace(4) null to ptr), ptr %{{.*}}
// LLVM: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr %{{.*}}
// LLVM: store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr %{{.*}}
// LLVM: store ptr addrspace(2) null, ptr %{{.*}}
// LLVM: store ptr addrspace(4) null, ptr %{{.*}}

private int *test_return_private(void) { return 0; }
// LLVM-LABEL: define {{.*}}ptr @test_return_private
// LLVM: ptr addrspacecast (ptr addrspace(4) null to ptr)

local int *test_return_local(void) { return (local int *)0; }
// LLVM-LABEL: define {{.*}}ptr addrspace(3) @test_return_local
// LLVM: ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3))

global int *test_return_global(void) { return NULL; }
// LLVM-LABEL: define {{.*}}ptr addrspace(1) @test_return_global
// LLVM: ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1))

constant int *test_return_constant(void) { return 0; }
// LLVM-LABEL: define {{.*}}ptr addrspace(2) @test_return_constant
// LLVM: ptr addrspace(2) null

generic int *test_return_generic(void) { return 0; }
// LLVM-LABEL: define {{.*}}ptr addrspace(4) @test_return_generic
// LLVM: ptr addrspace(4) null

local int *test_return_as_conversion(void) {
  return (local int *)(generic int *)(global int *)0;
}
// CIR-LABEL: cir.func {{.*}}@test_return_as_conversion
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR: cir.cast address_space %[[N]] : !cir.ptr<!s32i, lang_address_space(offload_generic)> -> !cir.ptr<!s32i, lang_address_space(offload_local)>

// LLVM-LABEL: define {{.*}}ptr addrspace(3) @test_return_as_conversion
// LLVM: ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3))

// SPV_INTEL_function_pointers forbids casting function pointers to generic.
typedef __attribute__((address_space(9))) void *FnPtrTy;
FnPtrTy test_return_fnptr(void) { return 0; }
// LLVM-LABEL: define {{.*}}ptr addrspace(9) @test_return_fnptr
// LLVM: ptr addrspace(9) null
