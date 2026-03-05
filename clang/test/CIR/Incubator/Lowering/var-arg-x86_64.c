// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-clangir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

#include <stdarg.h>

double f1(int n, ...) {
  va_list valist;
  va_start(valist, n);
  double res = va_arg(valist, double);
  va_end(valist);
  return res;
}

// CHECK: [[VA_LIST_TYPE:%.+]] = type { i32, i32, ptr, ptr }

// CHECK: define {{.*}}@f1
// CHECK: [[VA_LIST_ALLOCA:%.+]] = alloca {{.*}}[[VA_LIST_TYPE]]
// CHECK: [[VA_LIST:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: call {{.*}}@llvm.va_start.p0(ptr [[VA_LIST]])
// CHECK: [[VA_LIST2:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: [[FP_OFFSET_P:%.+]] = getelementptr {{.*}} [[VA_LIST2]], i32 0, i32 1
// CHECK: [[FP_OFFSET:%.+]] = load {{.*}}, ptr [[FP_OFFSET_P]]
// CHECK: [[COMPARED:%.+]] = icmp ule i32 {{.*}}, 160
// CHECK: br i1 [[COMPARED]], label %[[THEN_BB:.+]], label %[[ELSE_BB:.+]]
//
// CHECK: [[THEN_BB]]:
// CHECK:   [[UPDATED_FP_OFFSET:%.+]] = add i32 [[FP_OFFSET]], 8
// CHECK:   store i32 [[UPDATED_FP_OFFSET]], ptr [[FP_OFFSET_P]]
// CHECK:   br label %[[CONT_BB:.+]]
//
// CHECK: [[ELSE_BB]]:
// CHECK:   [[OVERFLOW_ARG_AREA_ADDR:%.+]] = getelementptr {{.*}} [[VA_LIST2]], i32 0, i32 2
// CHECK:   [[OVERFLOW_ARG_AREA:%.+]] = load ptr, ptr [[OVERFLOW_ARG_AREA_ADDR]]
// CHECK:   [[OVERFLOW_ARG_AREA_OFFSET:%.+]] = getelementptr {{.*}} [[OVERFLOW_ARG_AREA]], i64 8
// CHECK:   store ptr [[OVERFLOW_ARG_AREA_OFFSET]], ptr [[OVERFLOW_ARG_AREA_ADDR]]
// CHECK:   br label %[[CONT_BB]]
//
// CHECK: [[CONT_BB]]:
// CHECK: [[VA_LIST3:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: call {{.*}}@llvm.va_end.p0(ptr [[VA_LIST3]])

// CIR: cir.func {{.*}} @f1
// CIR: [[VA_LIST_ALLOCA:%.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>,
// CIR: [[RES:%.+]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["res",
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast array_to_ptrdecay [[VA_LIST_ALLOCA]]
// CIR: cir.va.start [[VASTED_VA_LIST]]
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast array_to_ptrdecay [[VA_LIST_ALLOCA]]
// CIR: [[VAARG_RESULT:%.+]] = cir.scope
// CIR: [[FP_OFFSET_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][1] {name = "fp_offset"}
// CIR: [[FP_OFFSET:%.+]] = cir.load [[FP_OFFSET_P]]
// CIR: [[OFFSET_CONSTANT:%.+]] = cir.const #cir.int<160>
// CIR: [[CMP:%.+]] = cir.cmp(le, [[FP_OFFSET]], [[OFFSET_CONSTANT]])
// CIR: cir.brcond [[CMP]] ^[[InRegBlock:.+]], ^[[InMemBlock:.+]] loc
//
// CIR: ^[[InRegBlock]]:
// CIR: [[REG_SAVE_AREA_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][3] {name = "reg_save_area"}
// CIR: [[REG_SAVE_AREA:%.+]] = cir.load [[REG_SAVE_AREA_P]]
// CIR: [[UPDATED:%.+]] = cir.ptr_stride [[REG_SAVE_AREA]], [[FP_OFFSET]] : (!cir.ptr<!void>, !u32i) -> !cir.ptr<!void>
// CIR: [[CONSTANT:%.+]] = cir.const #cir.int<8>
// CIR: [[ADDED:%.+]] = cir.binop(add, [[FP_OFFSET]], [[CONSTANT]])
// CIR: cir.store{{.*}} [[ADDED]], [[FP_OFFSET_P]]
// CIR: cir.br ^[[ContBlock:.+]]([[UPDATED]]
//
// CIR: ^[[InMemBlock]]:
// CIR: [[OVERFLOW_ARG_AREA_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][2] {name = "overflow_arg_area"}
// CIR: [[OVERFLOW_ARG_AREA:%.+]] = cir.load [[OVERFLOW_ARG_AREA_P]]
// CIR: [[OFFSET:%.+]] = cir.const #cir.int<8>
// CIR: [[CASTED:%.+]] = cir.cast bitcast [[OVERFLOW_ARG_AREA]] : !cir.ptr<!void>
// CIR: [[NEW_VALUE:%.+]] = cir.ptr_stride [[CASTED]], [[OFFSET]] : (!cir.ptr<!s8i>, !s32i) -> !cir.ptr<!s8i>
// CIR: [[CASTED_P:%.+]] = cir.cast bitcast [[OVERFLOW_ARG_AREA_P]] : !cir.ptr<!cir.ptr<!void>>
// CIR: cir.store [[NEW_VALUE]], [[CASTED_P]]
// CIR: cir.br ^[[ContBlock]]([[OVERFLOW_ARG_AREA]]
//
// CIR: ^[[ContBlock]]([[ARG:.+]]: !cir.ptr
// CIR: [[CASTED_ARG_P:%.+]] = cir.cast bitcast [[ARG]]
// CIR: [[CASTED_ARG:%.+]] = cir.load align(16) [[CASTED_ARG_P]]
// CIR: cir.yield [[CASTED_ARG]]
//
// CIR: cir.store{{.*}} [[VAARG_RESULT]], [[RES]]
long double f2(int n, ...) {
  va_list valist;
  va_start(valist, n);
  long double res = va_arg(valist, long double);
  va_end(valist);
  return res;
}

// CHECK: define {{.*}}@f2
// CHECK: [[RESULT:%.+]] = alloca x86_fp80
// CHECK: [[VA_LIST_ALLOCA:%.+]] = alloca {{.*}}[[VA_LIST_TYPE]]
// CHECK: [[RES:%.+]] = alloca x86_fp80
// CHECK: [[VA_LIST:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: call {{.*}}@llvm.va_start.p0(ptr [[VA_LIST]])
// CHECK: [[VA_LIST2:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: [[OVERFLOW_AREA_P:%.+]] = getelementptr {{.*}} [[VA_LIST2]], i32 0, i32 2
// CHECK: [[OVERFLOW_AREA:%.+]] = load ptr, ptr [[OVERFLOW_AREA_P]]
// Ptr Mask Operations
// CHECK: [[OVERFLOW_AREA_OFFSET_ALIGNED:%.+]] = getelementptr i8, ptr [[OVERFLOW_AREA]], i64 15
// CHECK: [[PTR_MASKED:%.+]] = call ptr @llvm.ptrmask.{{.*}}.[[PTR_SIZE_INT:.*]](ptr [[OVERFLOW_AREA_OFFSET_ALIGNED]], [[PTR_SIZE_INT]] -16)
// CHECK: [[OVERFLOW_AREA_NEXT:%.+]] = getelementptr i8, ptr [[PTR_MASKED]], i64 16
// CHECK: store ptr [[OVERFLOW_AREA_NEXT]], ptr [[OVERFLOW_AREA_P]]
// CHECK: [[VALUE:%.+]] = load x86_fp80, ptr [[PTR_MASKED]]
// CHECK: store x86_fp80 [[VALUE]], ptr [[RES]]
// CHECK: [[VA_LIST2:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: call {{.*}}@llvm.va_end.p0(ptr [[VA_LIST2]])
// CHECK: [[VALUE2:%.+]] = load x86_fp80, ptr [[RES]]
// CHECK: store x86_fp80 [[VALUE2]], ptr [[RESULT]]
// CHECK: [[RETURN_VALUE:%.+]] = load x86_fp80, ptr [[RESULT]]
// CHECK: ret x86_fp80 [[RETURN_VALUE]]

// CIR: cir.func {{.*}} @f2
// CIR: [[VA_LIST_ALLOCA:%.+]] = cir.alloca !cir.array<!rec___va_list_tag x 1>, !cir.ptr<!cir.array<!rec___va_list_tag x 1>>, ["valist"]
// CIR: [[RES:%.+]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["res"
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast array_to_ptrdecay [[VA_LIST_ALLOCA]]
// CIR: cir.va.start [[VASTED_VA_LIST]]
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast array_to_ptrdecay [[VA_LIST_ALLOCA]]
// CIR: [[OVERFLOW_AREA_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][2] {name = "overflow_arg_area"}
// CIR-DAG: [[OVERFLOW_AREA:%.+]] = cir.load [[OVERFLOW_AREA_P]]
// CIR-DAG: [[CASTED:%.+]] = cir.cast bitcast [[OVERFLOW_AREA]] : !cir.ptr<!void>
// CIR-DAG: [[CONSTANT:%.+]] = cir.const #cir.int<15>
// CIR-DAG: [[PTR_STRIDE:%.+]] = cir.ptr_stride [[CASTED]], [[CONSTANT]] : (!cir.ptr<!u8i>, !u32i) -> !cir.ptr<!u8i>
// CIR-DAG: [[MINUS_ALIGN:%.+]] = cir.const #cir.int<-16>
// CIR-DAG: [[ALIGNED:%.+]] = cir.ptr_mask([[PTR_STRIDE]], [[MINUS_ALIGN]]
// CIR: [[ALIGN:%.+]] = cir.const #cir.int<16>
// CIR: [[CAST_ALIGNED:%.+]] = cir.cast bitcast [[ALIGNED]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.long_double<!cir.f80>>
// CIR: [[CAST_ALIGNED_VALUE:%.+]] = cir.load [[CAST_ALIGNED]]
// CIR: cir.store{{.*}} [[CAST_ALIGNED_VALUE]], [[RES]]
// CIR. cir.via.end

const char *f3(va_list args) {
  return va_arg(args, const char *);
}

// CHECK: define{{.*}} @f3
// CHECK: [[VA_LIST_ALLOCA:%.+]] = alloca ptr
// ...
// CHECK: [[VA_LIST:%.+]] = load {{.*}} [[VA_LIST_ALLOCA]]
// CHECK: [[OFFSET_PTR:%.+]] = getelementptr {{.*}} [[VA_LIST]], i32 0, i32 0
// CHECK: [[OFFSET:%.+]] = load {{.*}}, ptr [[OFFSET_PTR]]
// CHECK: [[CMP:%.+]] = icmp ule i32 [[OFFSET]], 40
// CHECK: br i1 [[CMP]], label %[[THEN_BB:.+]], label %[[ELSE_BB:.+]]
//
// CHECK: [[THEN_BB]]:
// ...
// CHECK:   [[NEW_OFFSET:%.+]] = add i32 [[OFFSET]], 8
// CHECK:   store i32 [[NEW_OFFSET]], ptr [[OFFSET_PTR]]
// CHECK:   br label %[[CONT_BB:.+]]
//
// CHECK: [[ELSE_BB]]:
// ...
// CHECK:   [[OVERFLOW_ARG_AREA_ADDR:%.+]] = getelementptr {{.*}} [[VA_LIST]], i32 0, i32 2
// CHECK:   [[OVERFLOW_ARG_AREA:%.+]] = load ptr, ptr [[OVERFLOW_ARG_AREA_ADDR]]
// CHECK:   [[OVERFLOW_ARG_AREA_OFFSET:%.+]] = getelementptr {{.*}} [[OVERFLOW_ARG_AREA]], i64 8
// CHECK:   store ptr [[OVERFLOW_ARG_AREA_OFFSET]], ptr [[OVERFLOW_ARG_AREA_ADDR]]
// CHECK:   br label %[[CONT_BB]]
//
// CHECK: [[CONT_BB]]:
// ...
// CHECK: ret

// CIR-LABEL:   cir.func {{.*}} @f3(
// CIR:           %[[VALIST_VAR:.*]] = cir.alloca !cir.ptr<!rec___va_list_tag>, !cir.ptr<!cir.ptr<!rec___va_list_tag>>, ["args", init] {alignment = 8 : i64}
// CIR:           %[[VALIST:.*]] = cir.load align(8) %[[VALIST_VAR]] : !cir.ptr<!cir.ptr<!rec___va_list_tag>>, !cir.ptr<!rec___va_list_tag>
// CIR:           %[[GP_OFFSET_PTR:.*]] = cir.get_member %[[VALIST]][0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:           %[[GP_OFFSET:.*]] = cir.load %[[GP_OFFSET_PTR]] : !cir.ptr<!u32i>, !u32i
// CIR:           %[[VAL_6:.*]] = cir.const #cir.int<40> : !u32i
// CIR:           %[[VAL_7:.*]] = cir.cmp(le, %[[GP_OFFSET]], %[[VAL_6]]) : !u32i, !cir.bool
// CIR:           cir.brcond %[[VAL_7]]

// CIR:           %[[REG_SAVE_AREA_PTR:.*]] = cir.get_member %[[VALIST]][3] {name = "reg_save_area"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!cir.ptr<!void>>
// CIR:           %[[REG_SAVE_AREA:.*]] = cir.load %[[REG_SAVE_AREA_PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:           %[[CUR_REG_SAVE_AREA:.*]] = cir.ptr_stride %[[REG_SAVE_AREA]], %[[GP_OFFSET]] : (!cir.ptr<!void>, !u32i) -> !cir.ptr<!void>
// CIR:           %[[VAL_11:.*]] = cir.const #cir.int<8> : !u32i
// CIR:           %[[NEW_REG_SAVE_AREA:.*]] = cir.binop(add, %[[GP_OFFSET]], %[[VAL_11]]) : !u32i
// CIR:           cir.store %[[NEW_REG_SAVE_AREA]], %[[GP_OFFSET_PTR]] : !u32i, !cir.ptr<!u32i>
// CIR:           cir.br ^[[CONT_BB:.*]](%[[CUR_REG_SAVE_AREA]] : !cir.ptr<!void>)

// CIR:           %[[OVERFLOW_ARG_AREA_PTR:.*]] = cir.get_member %[[VALIST]][2] {name = "overflow_arg_area"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!cir.ptr<!void>>
// CIR:           %[[OVERFLOW_ARG_AREA:.*]] = cir.load %[[OVERFLOW_ARG_AREA_PTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:           %[[VAL_15:.*]] = cir.const #cir.int<8> : !s32i
// CIR:           %[[CUR_OVERFLOW_ARG_AREA:.*]] = cir.cast bitcast %[[OVERFLOW_ARG_AREA]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
// CIR:           %[[NEW_OVERFLOW_ARG_AREA:.*]] = cir.ptr_stride %[[CUR_OVERFLOW_ARG_AREA]], %[[VAL_15]] : (!cir.ptr<!s8i>, !s32i) -> !cir.ptr<!s8i>
// CIR:           %[[VAL_18:.*]] = cir.cast bitcast %[[OVERFLOW_ARG_AREA_PTR]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.ptr<!s8i>>
// CIR:           cir.store %[[NEW_OVERFLOW_ARG_AREA]], %[[VAL_18]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR:           cir.br ^[[CONT_BB]](%[[OVERFLOW_ARG_AREA]] : !cir.ptr<!void>)

// ...
// CIR:           cir.return

void f4(va_list args) {
  for (; va_arg(args, int); );
}
// CIR-LABEL:   cir.func {{.*}} @f4
// CIR:           cir.for : cond {
// CIR:             %[[VALIST:.*]] = cir.load align(8) %[[VALIST_VAR]] : !cir.ptr<!cir.ptr<!rec___va_list_tag>>, !cir.ptr<!rec___va_list_tag>
// CIR:             %[[VAARG_RESULT:.*]] = cir.scope {
//                    ... // The contents are tested elsewhere.
// CIR:               cir.yield {{.*}} : !s32i
// CIR:             } : !s32i
// CIR:             %[[CMP:.*]] = cir.cast int_to_bool %[[VAARG_RESULT]] : !s32i -> !cir.bool
// CIR:             cir.condition(%[[CMP]])
// CIR:           } body {
// CIR:             cir.yield
// CIR:           } step {
// CIR:             cir.yield
// CIR:           }
// CIR:           cir.return
// CIR:         }
