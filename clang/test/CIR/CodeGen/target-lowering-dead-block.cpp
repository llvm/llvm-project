// RUN: %clang_cc1 -fcuda-is-device -triple amdgpu -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck --check-prefix=CIR,CIR-PRE --input-file=%t.pre.cir %s
// RUN: FileCheck --check-prefix=CIR,CIR-POST --input-file=%t.cir %s

__attribute__((device)) int test(int input) {
  int a = input;
  if (!input) {
    __builtin_trap();
    return a;
  } else {
    return a;
  }
}

// CIR-LABEL: cir.func {{.*}}@_Z4testi

// CIR-PRE:       %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR-POST:      %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i, target_address_space(5)>

// CIR-PRE:       %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR-POST:       %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i, target_address_space(5)>

// CIR-POST-NOT:  builtin.unrealized_conversion_cast

// CIR-PRE:       %[[ACAST:.*]] = cir.cast address_space %[[A]] : !cir.ptr<!s32i, lang_address_space(offload_private)> -> !cir.ptr<!s32i>
// CIR-POST:      %[[ACAST:.*]] = cir.cast address_space %[[A]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>

// CIR:           cir.trap
// CIR:           %[[LOAD:.*]] = cir.load align(4) %[[ACAST]] : !cir.ptr<!s32i>, !s32i
// CIR-PRE-NEXT:  cir.store %[[LOAD]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR-PRE-NEXT:  %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i, lang_address_space(offload_private)>, !s32i
// CIR-POST-NEXT: cir.store %[[LOAD]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR-POST-NEXT: %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i, target_address_space(5)>, !s32i
// CIR-NEXT:      cir.return %[[RET]] : !s32i
