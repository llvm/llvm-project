// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir  -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck -check-prefix=BEFORE %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir  -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck -check-prefix=AFTER %s

class E {
  public:
    E();
    ~E();
};

void t_new_constant_size_constructor() {
  auto p = new E[3];
}

// BEFORE:  cir.func @_Z31t_new_constant_size_constructorv
// BEFORE:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<3> : !u64i
// BEFORE:    %[[SIZE_WITHOUT_COOKIE:.*]] = cir.const #cir.int<3> : !u64i
// BEFORE:    %[[ALLOC_SIZE:.*]] = cir.const #cir.int<11> : !u64i
// BEFORE:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]])
// BEFORE:    %[[COOKIE_PTR:.*]] = cir.cast(bitcast, %[[ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u64i>
// BEFORE:    cir.store %[[NUM_ELEMENTS]], %[[COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// BEFORE:    %[[PTR_AS_U8:.*]] = cir.cast(bitcast, %[[ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u8i>
// BEFORE:    %[[OFFSET:.*]] = cir.const #cir.int<8> : !s32i
// BEFORE:    %[[OBJ_PTR:.*]] = cir.ptr_stride(%[[PTR_AS_U8]] : !cir.ptr<!u8i>, %[[OFFSET]] : !s32i), !cir.ptr<!u8i>
// BEFORE:    %[[OBJ_ELEM_PTR:.*]] = cir.cast(bitcast, %[[OBJ_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!ty_E>
// BEFORE:    %[[OBJ_ARRAY_PTR:.*]] = cir.cast(bitcast, %[[OBJ_ELEM_PTR]] : !cir.ptr<!ty_E>), !cir.ptr<!cir.array<!ty_E x 3>>
// BEFORE:    cir.array.ctor(%[[OBJ_ARRAY_PTR]] : !cir.ptr<!cir.array<!ty_E x 3>>) {
// BEFORE:    ^bb0(%arg0: !cir.ptr<!ty_E>
// BEFORE:      cir.call @_ZN1EC1Ev(%arg0) : (!cir.ptr<!ty_E>) -> ()
// BEFORE:      cir.yield
// BEFORE:    }

// AFTER:  cir.func @_Z31t_new_constant_size_constructorv
// AFTER:    %[[NUM_ELEMENTS:.*]] = cir.const #cir.int<3> : !u64i
// AFTER:    %[[SIZE_WITHOUT_COOKIE:.*]] = cir.const #cir.int<3> : !u64i
// AFTER:    %[[ALLOC_SIZE:.*]] = cir.const #cir.int<11> : !u64i
// AFTER:    %[[ALLOC_PTR:.*]] = cir.call @_Znam(%[[ALLOC_SIZE]])
// AFTER:    %[[COOKIE_PTR:.*]] = cir.cast(bitcast, %[[ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u64i>
// AFTER:    cir.store %[[NUM_ELEMENTS]], %[[COOKIE_PTR]] : !u64i, !cir.ptr<!u64i>
// AFTER:    %[[PTR_AS_U8:.*]] = cir.cast(bitcast, %[[ALLOC_PTR]] : !cir.ptr<!void>), !cir.ptr<!u8i>
// AFTER:    %[[OFFSET:.*]] = cir.const #cir.int<8> : !s32i
// AFTER:    %[[OBJ_PTR:.*]] = cir.ptr_stride(%[[PTR_AS_U8]] : !cir.ptr<!u8i>, %[[OFFSET]] : !s32i), !cir.ptr<!u8i>
// AFTER:    %[[OBJ_ELEM_PTR:.*]] = cir.cast(bitcast, %[[OBJ_PTR]] : !cir.ptr<!u8i>), !cir.ptr<!ty_E>
// AFTER:    %[[OBJ_ARRAY_PTR:.*]] = cir.cast(bitcast, %[[OBJ_ELEM_PTR]] : !cir.ptr<!ty_E>), !cir.ptr<!cir.array<!ty_E x 3>>
// AFTER:    %[[NUM_ELEMENTS2:.*]] = cir.const #cir.int<3> : !u64i
// AFTER:    %[[ELEM_PTR:.*]] = cir.cast(array_to_ptrdecay, %10 : !cir.ptr<!cir.array<!ty_E x 3>>), !cir.ptr<!ty_E>
// AFTER:    %[[END_PTR:.*]] = cir.ptr_stride(%[[ELEM_PTR]] : !cir.ptr<!ty_E>, %[[NUM_ELEMENTS2]] : !u64i), !cir.ptr<!ty_E>
// AFTER:    %[[CUR_ELEM_ALLOCA:.*]] = cir.alloca !cir.ptr<!ty_E>, !cir.ptr<!cir.ptr<!ty_E>>, ["__array_idx"] {alignment = 1 : i64}
// AFTER:    cir.store %[[ELEM_PTR]], %[[CUR_ELEM_ALLOCA]] : !cir.ptr<!ty_E>, !cir.ptr<!cir.ptr<!ty_E>>
// AFTER:    cir.do {
// AFTER:      %[[CUR_ELEM_PTR:.*]] = cir.load %[[CUR_ELEM_ALLOCA]] : !cir.ptr<!cir.ptr<!ty_E>>, !cir.ptr<!ty_E>
// AFTER:      %[[OFFSET:.*]] = cir.const #cir.int<1> : !u64i
// AFTER:      cir.call @_ZN1EC1Ev(%[[CUR_ELEM_PTR]]) : (!cir.ptr<!ty_E>) -> ()
// AFTER:      %[[NEXT_PTR:.*]] = cir.ptr_stride(%[[CUR_ELEM_PTR]] : !cir.ptr<!ty_E>, %[[OFFSET]] : !u64i), !cir.ptr<!ty_E>
// AFTER:      cir.store %[[NEXT_PTR]], %[[CUR_ELEM_ALLOCA]] : !cir.ptr<!ty_E>, !cir.ptr<!cir.ptr<!ty_E>>
// AFTER:      cir.yield
// AFTER:    } while {
// AFTER:      %[[CUR_ELEM_PTR2:.*]] = cir.load %[[CUR_ELEM_ALLOCA]] : !cir.ptr<!cir.ptr<!ty_E>>, !cir.ptr<!ty_E>
// AFTER:      %[[END_TEST:.*]] = cir.cmp(eq, %[[CUR_ELEM_PTR2]], %[[END_PTR]]) : !cir.ptr<!ty_E>, !cir.bool
// AFTER:      cir.condition(%[[END_TEST]])
// AFTER:    }
