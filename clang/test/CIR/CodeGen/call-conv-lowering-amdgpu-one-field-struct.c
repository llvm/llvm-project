// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -cir-call-conv-lowering=target=amdgpu -o %t-ccl.cir
// RUN: FileCheck --input-file=%t-ccl.cir %s --check-prefix=CIR
// RUN: cir-translate %t-ccl.cir -cir-to-llvmir -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// A struct whose only field is an aggregate is passed as that field.

typedef struct { int i[4]; } single_array_element_struct_arg_t;
typedef struct { _Complex double c; } cdwrap_t;
typedef struct { struct { int a, b, c; } t; } nestwrap_t;
typedef struct { int a, b, c; } triple_t;

// CIR: ![[ANON:rec_anon[^ ]*]] = !cir.struct<"anon{{[^"]*}}" {data !s32i, data !s32i, data !s32i}>
// CIR: !rec_nestwrap_t = !cir.struct<"nestwrap_t" {data ![[ANON]]}>

// The array field is passed in place of the struct.
void func_single_array_element_struct_arg(single_array_element_struct_arg_t arg1) {}

// CIR:      cir.func{{.*}} @func_single_array_element_struct_arg(%[[ARG:.*]]: !cir.array<!s32i x 4>)
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)>
// CIR-NEXT:   %[[FIELD:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)> -> !cir.ptr<!cir.array<!s32i x 4>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[ARG]], %[[FIELD]] : !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>, target_address_space(5)>
// CIR-NEXT:   %[[ARG1:.*]] = cir.alloca "arg1" align(4) init : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)>
// CIR-NEXT:   %[[ARG1_VIEW:.*]] = cir.cast address_space %[[ARG1]] : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)> -> !cir.ptr<!rec_single_array_element_struct_arg_t>
// CIR-NEXT:   cir.copy %[[COERCE]] align(4) to %[[ARG1_VIEW]] align(4) : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)>, !cir.ptr<!rec_single_array_element_struct_arg_t>
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @func_single_array_element_struct_arg([4 x i32] %{{.*}})

// The complex field is passed in place of the struct.
void arg_cdwrap(cdwrap_t w) {}

// CIR:      cir.func{{.*}} @arg_cdwrap(%[[ARG:.*]]: !cir.complex<!cir.double>)
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[FIELD:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<!rec_cdwrap_t, target_address_space(5)> -> !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[ARG]], %[[FIELD]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// CIR-NEXT:   %[[W:.*]] = cir.alloca "w" align(8) init : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W_VIEW:.*]] = cir.cast address_space %[[W]] : !cir.ptr<!rec_cdwrap_t, target_address_space(5)> -> !cir.ptr<!rec_cdwrap_t>
// CIR-NEXT:   cir.copy %[[COERCE]] align(8) to %[[W_VIEW]] align(8) : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>, !cir.ptr<!rec_cdwrap_t>
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @arg_cdwrap({ double, double } %{{.*}})

// The inner struct is passed whole, not split into its three ints.
void arg_nestwrap(nestwrap_t w) {}

// CIR:      cir.func{{.*}} @arg_nestwrap(%[[ARG:.*]]: ![[ANON]])
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_nestwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[FIELD:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<!rec_nestwrap_t, target_address_space(5)> -> !cir.ptr<![[ANON]], target_address_space(5)>
// CIR-NEXT:   cir.store %[[ARG]], %[[FIELD]] : ![[ANON]], !cir.ptr<![[ANON]], target_address_space(5)>
// CIR-NEXT:   %[[W:.*]] = cir.alloca "w" align(4) init : !cir.ptr<!rec_nestwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W_VIEW:.*]] = cir.cast address_space %[[W]] : !cir.ptr<!rec_nestwrap_t, target_address_space(5)> -> !cir.ptr<!rec_nestwrap_t>
// CIR-NEXT:   cir.copy %[[COERCE]] align(4) to %[[W_VIEW]] align(4) : !cir.ptr<!rec_nestwrap_t, target_address_space(5)>, !cir.ptr<!rec_nestwrap_t>
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @arg_nestwrap(%struct.anon{{.*}} %{{.*}})

// A struct of three ints is split into three arguments.
void arg_triple(triple_t w) {}

// CIR:      cir.func{{.*}} @arg_triple(%[[A:[^:]+]]: !s32i, %[[B:[^:]+]]: !s32i, %[[C:[^:]+]]: !s32i)
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_triple_t, target_address_space(5)>
// CIR-NEXT:   %[[A_PTR:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<!rec_triple_t, target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[A]], %[[A_PTR]] : !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[B_PTR:.*]] = cir.get_member %[[COERCE]][1] {name = ""} : !cir.ptr<!rec_triple_t, target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[B]], %[[B_PTR]] : !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[C_PTR:.*]] = cir.get_member %[[COERCE]][2] {name = ""} : !cir.ptr<!rec_triple_t, target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[C]], %[[C_PTR]] : !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR-NEXT:   %[[W:.*]] = cir.alloca "w" align(4) init : !cir.ptr<!rec_triple_t, target_address_space(5)>
// CIR-NEXT:   %[[W_VIEW:.*]] = cir.cast address_space %[[W]] : !cir.ptr<!rec_triple_t, target_address_space(5)> -> !cir.ptr<!rec_triple_t>
// CIR-NEXT:   cir.copy %[[COERCE]] align(4) to %[[W_VIEW]] align(4) : !cir.ptr<!rec_triple_t, target_address_space(5)>, !cir.ptr<!rec_triple_t>
// CIR-NEXT:   cir.return

// LLVM: define dso_local void @arg_triple(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

// The caller passes the array field.
void call_single_array(single_array_element_struct_arg_t a) {
  func_single_array_element_struct_arg(a);
}

// CIR:      cir.func{{.*}} @call_single_array(%[[ARG:.*]]: !cir.array<!s32i x 4>)
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)>
// CIR-NEXT:   %[[FIELD:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)> -> !cir.ptr<!cir.array<!s32i x 4>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[ARG]], %[[FIELD]] : !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>, target_address_space(5)>
// CIR-NEXT:   %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)>
// CIR-NEXT:   %[[A_VIEW:.*]] = cir.cast address_space %[[A]] : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)> -> !cir.ptr<!rec_single_array_element_struct_arg_t>
// CIR-NEXT:   cir.copy %[[COERCE]] align(4) to %[[A_VIEW]] align(4) : !cir.ptr<!rec_single_array_element_struct_arg_t, target_address_space(5)>, !cir.ptr<!rec_single_array_element_struct_arg_t>
// CIR-NEXT:   %[[A_FIELD:.*]] = cir.get_member %[[A_VIEW]][0] {name = ""} : !cir.ptr<!rec_single_array_element_struct_arg_t> -> !cir.ptr<!cir.array<!s32i x 4>>
// CIR-NEXT:   %[[A_ARR:.*]] = cir.load %[[A_FIELD]] : !cir.ptr<!cir.array<!s32i x 4>>, !cir.array<!s32i x 4>
// CIR-NEXT:   cir.call @func_single_array_element_struct_arg(%[[A_ARR]]) {convergent} : (!cir.array<!s32i x 4>) -> ()
// CIR-NEXT:   cir.return

// LLVM:      define dso_local void @call_single_array([4 x i32] %{{.*}})
// LLVM:        %[[A_ARR:.*]] = load [4 x i32], ptr %{{.*}}, align 4
// LLVM-NEXT:   call void @func_single_array_element_struct_arg([4 x i32] %[[A_ARR]])

// The argument is passed as its field, but the return stays the struct.
cdwrap_t ret_cdwrap(cdwrap_t w) { return w; }

// CIR:      cir.func{{.*}} @ret_cdwrap(%[[ARG:.*]]: !cir.complex<!cir.double>) -> !rec_cdwrap_t
// CIR-NEXT:   %[[COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[FIELD:.*]] = cir.get_member %[[COERCE]][0] {name = ""} : !cir.ptr<!rec_cdwrap_t, target_address_space(5)> -> !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[ARG]], %[[FIELD]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" align(8) : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W:.*]] = cir.alloca "w" align(8) init : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W_VIEW:.*]] = cir.cast address_space %[[W]] : !cir.ptr<!rec_cdwrap_t, target_address_space(5)> -> !cir.ptr<!rec_cdwrap_t>
// CIR-NEXT:   %[[RETVAL_VIEW:.*]] = cir.cast address_space %[[RETVAL]] : !cir.ptr<!rec_cdwrap_t, target_address_space(5)> -> !cir.ptr<!rec_cdwrap_t>
// CIR-NEXT:   cir.copy %[[COERCE]] align(8) to %[[W_VIEW]] align(8) : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>, !cir.ptr<!rec_cdwrap_t>
// CIR-NEXT:   cir.copy %[[W_VIEW]] align(8) to %[[RETVAL_VIEW]] align(8) : !cir.ptr<!rec_cdwrap_t>
// CIR-NEXT:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>, !rec_cdwrap_t
// CIR-NEXT:   cir.return %[[RET]] : !rec_cdwrap_t

// LLVM: define dso_local %struct.cdwrap_t @ret_cdwrap({ double, double } %{{.*}})
