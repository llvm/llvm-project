
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -cl-std=CL2.0 -fclangir \
// RUN:   -emit-cir -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t-amdgcn.cir 2> %t-amdgcn.pre.cir
// RUN: FileCheck %s --check-prefixes=CIR,CIR-AMDGCN \
// RUN:   --input-file=%t-amdgcn.pre.cir
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -cl-std=CL2.0 -fclangir \
// RUN:   -emit-llvm -O0 %s -o %t-amdgcn-cir.ll
// RUN: FileCheck %s --check-prefixes=LLVM,LLVM-AMDGCN \
// RUN:   --input-file=%t-amdgcn-cir.ll
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -cl-std=CL2.0 \
// RUN:   -emit-llvm -O0 %s -o %t-amdgcn.ll
// RUN: FileCheck %s --check-prefixes=LLVM,LLVM-AMDGCN --input-file=%t-amdgcn.ll

#define NULL ((void *)0)

void cast_bool_private(private char* p) {
  if (p)
    *p = 0;
}

// CIR-LABEL: cir.func {{.*}}@cast_bool_private
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_private)>
// CIR: cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_private)>

// LLVM-LABEL: define {{.*}}void @cast_bool_private
// LLVM-AMDGCN: icmp ne ptr addrspace(5) %{{.*}}, addrspacecast (ptr null to ptr addrspace(5))

void cast_bool_local(local char* p) {
  if (p)
    *p = 0;
}

// CIR-LABEL: cir.func {{.*}}@cast_bool_local
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_local)>
// CIR: cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_local)>

// LLVM-LABEL: define {{.*}}void @cast_bool_local
// LLVM-AMDGCN: icmp ne ptr addrspace(3) %{{.*}}, addrspacecast (ptr null to ptr addrspace(3))

void cast_bool_global(global char* p) {
  if (p)
    *p = 0;
}

// CIR-LABEL: cir.func {{.*}}@cast_bool_global
// CIR-AMDGCN: cir.cast ptr_to_bool %{{.*}} : !cir.ptr<!s8i, lang_address_space(offload_global)> -> !cir.bool

// LLVM-LABEL: define {{.*}}void @cast_bool_global
// LLVM-AMDGCN: icmp ne ptr addrspace(1) %{{.*}}, null

char cast_bool_constant(constant char* p) {
  if (p)
    return *p;
  else
    return 0;
}

// CIR-LABEL: cir.func {{.*}}@cast_bool_constant
// CIR: cir.cast ptr_to_bool %{{.*}} : !cir.ptr<!s8i, lang_address_space(offload_constant)> -> !cir.bool

// LLVM-LABEL: define {{.*}}i8 @cast_bool_constant
// LLVM-AMDGCN: icmp ne ptr addrspace(4) %{{.*}}, null

void cast_bool_generic(generic char* p) {
  if (p)
    *p = 0;
}

// CIR-LABEL: cir.func {{.*}}@cast_bool_generic
// CIR: cir.cast ptr_to_bool %{{.*}} : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.bool

// LLVM-LABEL: define {{.*}}void @cast_bool_generic
// LLVM-AMDGCN: icmp ne ptr %{{.*}}, null

bool cast_bool_implicit_local(local char* p) {
  return p;
}

// CIR-LABEL: cir.func {{.*}}@cast_bool_implicit_local
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_local)>
// CIR: cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_local)>

// LLVM-LABEL: define {{.*}}i1 @cast_bool_implicit_local
// LLVM-AMDGCN: icmp ne ptr addrspace(3) %{{.*}}, addrspacecast (ptr null to ptr addrspace(3))

int test_not_private_ptr(private char* p) {
  return !p;
}

// CIR-LABEL: cir.func {{.*}}@test_not_private_ptr
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_private)>
// CIR: %[[CMP:.*]] = cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_private)>
// CIR: cir.not %[[CMP]] : !cir.bool

// LLVM-LABEL: define {{.*}}i32 @test_not_private_ptr
// LLVM-AMDGCN: %[[TOBOOL:.*]] = icmp ne ptr addrspace(5) %{{.*}}, addrspacecast (ptr null to ptr addrspace(5))
// LLVM: %[[LNOT:.*]] = xor i1 %[[TOBOOL]], true
// LLVM: zext i1 %[[LNOT]] to i32

int test_not_local_ptr(local char* p) {
  return !p;
}

// CIR-LABEL: cir.func {{.*}}@test_not_local_ptr
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_local)>
// CIR: %[[CMP:.*]] = cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_local)>
// CIR: cir.not %[[CMP]] : !cir.bool

// LLVM-LABEL: define {{.*}}i32 @test_not_local_ptr
// LLVM-AMDGCN: %[[TOBOOL:.*]] = icmp ne ptr addrspace(3) %{{.*}}, addrspacecast (ptr null to ptr addrspace(3))
// LLVM: %[[LNOT:.*]] = xor i1 %[[TOBOOL]], true
// LLVM: zext i1 %[[LNOT]] to i32

int test_and_ptr(private char* p1, local char* p2) {
  return p1 && p2;
}

// CIR-LABEL: cir.func {{.*}}@test_and_ptr
// CIR: %[[LHS:.*]] = cir.cmp ne %{{.*}}, %{{.*}} : !cir.ptr<!s8i, lang_address_space(offload_private)>
// CIR: cir.ternary(%[[LHS]], true {
// CIR: cir.cmp ne %{{.*}}, %{{.*}} : !cir.ptr<!s8i, lang_address_space(offload_local)>

// LLVM-LABEL: define {{.*}}i32 @test_and_ptr
// LLVM-AMDGCN: icmp ne ptr addrspace(5) %{{.*}}, addrspacecast (ptr null to ptr addrspace(5))
// LLVM-AMDGCN: %[[RHS:.*]] = icmp ne ptr addrspace(3) %{{.*}}, addrspacecast (ptr null to ptr addrspace(3))
// LLVM: %[[AND:.*]] = phi i1 [ false, %{{.*}} ], [ %[[RHS]], %{{.*}} ]
// LLVM: zext i1 %[[AND]] to i32

void cmp_local(local char* p) {
  if (p != 0)
    *p = 0;
}

// CIR-LABEL: cir.func {{.*}}@cmp_local
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_local)>
// CIR: cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_local)>

// LLVM-LABEL: define {{.*}}void @cmp_local
// LLVM-AMDGCN: icmp ne ptr addrspace(3) %{{.*}}, addrspacecast (ptr null to ptr addrspace(3))

void cmp_NULL_private(private char* p) {
  if (p != NULL)
    *p = 0;
}

// CIR-LABEL: cir.func {{.*}}@cmp_NULL_private
// CIR: %[[N:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s8i, lang_address_space(offload_generic)>
// CIR: %[[NP:.*]] = cir.cast address_space %[[N]] : !cir.ptr<!s8i, lang_address_space(offload_generic)> -> !cir.ptr<!s8i, lang_address_space(offload_private)>
// CIR: cir.cmp ne %{{.*}}, %[[NP]] : !cir.ptr<!s8i, lang_address_space(offload_private)>

// LLVM-LABEL: define {{.*}}void @cmp_NULL_private
// LLVM-AMDGCN: icmp ne ptr addrspace(5) %{{.*}}, addrspacecast (ptr null to ptr addrspace(5))
