// Tests for LangAddressSpaceAttr -> TargetAddressSpaceAttr conversion in the
// cir-target-lowering pass for the AMDGPU target. Exercises:
//   - CIRGlobalOpTargetLowering       (GlobalOp addr_space attribute)
//   - CIRFuncOpTargetLowering         (FuncOp pointer argument types)
//   - CIRGenericTargetLoweringPattern (get_global result pointer types)
//
// AMDGPU address space mapping:
//   offload_global   -> 1  (GLOBAL_ADDRESS)
//   offload_local    -> 3  (LOCAL_ADDRESS)
//   offload_constant -> 4  (CONSTANT_ADDRESS)
//   offload_private  -> 5  (PRIVATE_ADDRESS)

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir \
// RUN:   -mmlir -mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck --check-prefix=PRE --input-file=%t.pre.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=POST --input-file=%t.cir %s

// ---- GlobalOp: lang AS emitted before target lowering, target AS after ----

// Note: opencl_local/opencl_private globals require loader_uninitialized which
// is NYI in CIR. Those AS are exercised via pointer parameter types below.

int [[clang::opencl_global]]   g_global   = 0;
int [[clang::opencl_constant]] g_constant = 0;

// PRE-DAG: cir.global {{.*}} lang_address_space(offload_global)   @g_global
// PRE-DAG: cir.global {{.*}} lang_address_space(offload_constant) @g_constant

// POST-DAG: cir.global {{.*}} target_address_space(1) @g_global
// POST-DAG: cir.global {{.*}} target_address_space(4) @g_constant

// ---- FuncOp: pointer args exercise all four lang AS -> target AS ----

void func_ptr_args(int [[clang::opencl_global]]   *global_ptr,
                   int [[clang::opencl_local]]    *local_ptr,
                   int [[clang::opencl_constant]] *const_ptr,
                   int [[clang::opencl_private]]  *private_ptr) {}

// PRE:  cir.func {{.*}} @_Z13func_ptr_args
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_global)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_local)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_constant)>
// PRE-SAME: !cir.ptr<!s32i, lang_address_space(offload_private)>

// POST: cir.func {{.*}} @_Z13func_ptr_args
// POST-SAME: !cir.ptr<!s32i, target_address_space(1)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(3)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(4)>
// POST-SAME: !cir.ptr<!s32i, target_address_space(5)>

// ---- get_global: result pointer type with lang AS -> target AS ----

void get_globals() {
  (void)g_global;
  (void)g_constant;
}

// PRE:  cir.func {{.*}} @_Z11get_globalsv
// PRE:    cir.get_global @g_global   : !cir.ptr<!s32i, lang_address_space(offload_global)>
// PRE:    cir.get_global @g_constant : !cir.ptr<!s32i, lang_address_space(offload_constant)>

// POST: cir.func {{.*}} @_Z11get_globalsv
// POST:   cir.get_global @g_global   : !cir.ptr<!s32i, target_address_space(1)>
// POST:   cir.get_global @g_constant : !cir.ptr<!s32i, target_address_space(4)>
