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

// CIR: cir.func @f1
// CIR: [[VA_LIST_ALLOCA:%.+]] = cir.alloca !cir.array<!ty___va_list_tag x 1>,
// CIR: [[RES:%.+]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["res",
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast(array_to_ptrdecay, [[VA_LIST_ALLOCA]] 
// CIR: cir.va.start [[VASTED_VA_LIST]]
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast(array_to_ptrdecay, [[VA_LIST_ALLOCA]] 
// CIR: [[FP_OFFSET_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][1] {name = "fp_offset"}
// CIR: [[FP_OFFSET:%.+]] = cir.load [[FP_OFFSET_P]]
// CIR: [[OFFSET_CONSTANT:%.+]] = cir.const #cir.int<160>
// CIR: [[CMP:%.+]] = cir.cmp(le, [[FP_OFFSET]], [[OFFSET_CONSTANT]])
// CIR: cir.brcond [[CMP]] ^[[InRegBlock:.+]], ^[[InMemBlock:.+]] loc
// 
// CIR: ^[[InRegBlock]]:
// CIR: [[REG_SAVE_AREA_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][3] {name = "reg_save_area"}
// CIR: [[REG_SAVE_AREA:%.+]] = cir.load [[REG_SAVE_AREA_P]]
// CIR: [[UPDATED:%.+]] = cir.ptr_stride([[REG_SAVE_AREA]] {{.*}}, [[FP_OFFSET]]
// CIR: [[CONSTANT:%.+]] = cir.const #cir.int<8>
// CIR: [[ADDED:%.+]] = cir.binop(add, [[FP_OFFSET]], [[CONSTANT]])
// CIR: cir.store [[ADDED]], [[FP_OFFSET_P]]
// CIR: cir.br ^[[ContBlock:.+]]([[UPDATED]]
//
// CIR: ^[[InMemBlock]]:
// CIR: [[OVERFLOW_ARG_AREA_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][2] {name = "overflow_arg_area"}
// CIR: [[OVERFLOW_ARG_AREA:%.+]] = cir.load [[OVERFLOW_ARG_AREA_P]]
// CIR: [[OFFSET:%.+]] = cir.const #cir.int<8>
// CIR: [[CASTED:%.+]] = cir.cast(bitcast, [[OVERFLOW_ARG_AREA]] : !cir.ptr<!void>)
// CIR: [[NEW_VALUE:%.+]] = cir.ptr_stride([[CASTED]] : !cir.ptr<!s8i>, [[OFFSET]]
// CIR: [[CASTED_P:%.+]] = cir.cast(bitcast, [[OVERFLOW_ARG_AREA_P]] : !cir.ptr<!cir.ptr<!void>>)
// CIR: store [[NEW_VALUE]], [[CASTED_P]]
// CIR: cir.br ^[[ContBlock]]([[OVERFLOW_ARG_AREA]]
//
// CIR: ^[[ContBlock]]([[ARG:.+]]: !cir.ptr
// CIR: [[CASTED_ARG_P:%.+]] = cir.cast(bitcast, [[ARG]]
// CIR: [[CASTED_ARG:%.+]] = cir.load align(16) [[CASTED_ARG_P]]
// CIR: store [[CASTED_ARG]], [[RES]]
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
// CHECK: [[OVERFLOW_AREA_OFFSET_ALIGNED_P:%.+]] = ptrtoint ptr [[OVERFLOW_AREA_OFFSET_ALIGNED]] to i32
// CHECK: [[MASKED:%.+]] = and i32 [[OVERFLOW_AREA_OFFSET_ALIGNED_P]], -16
// CHECK: [[DIFF:%.+]] = sub i32 [[OVERFLOW_AREA_OFFSET_ALIGNED_P]], [[MASKED]]
// CHECK: [[PTR_MASKED:%.+]] = getelementptr i8, ptr [[OVERFLOW_AREA_OFFSET_ALIGNED]], i32 [[DIFF]]
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

// CIR: cir.func @f2
// CIR: [[VA_LIST_ALLOCA:%.+]] = cir.alloca !cir.array<!ty___va_list_tag x 1>, !cir.ptr<!cir.array<!ty___va_list_tag x 1>>, ["valist"]
// CIR: [[RES:%.+]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["res"
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast(array_to_ptrdecay, [[VA_LIST_ALLOCA]] 
// CIR: cir.va.start [[VASTED_VA_LIST]]
// CIR: [[VASTED_VA_LIST:%.+]] = cir.cast(array_to_ptrdecay, [[VA_LIST_ALLOCA]] 
// CIR: [[OVERFLOW_AREA_P:%.+]] = cir.get_member [[VASTED_VA_LIST]][2] {name = "overflow_arg_area"}
// CIR-DAG: [[OVERFLOW_AREA:%.+]] = cir.load [[OVERFLOW_AREA_P]]
// CIR-DAG: [[CASTED:%.+]] = cir.cast(bitcast, [[OVERFLOW_AREA]] : !cir.ptr<!void>)
// CIR-DAG: [[CONSTANT:%.+]] = cir.const #cir.int<15>
// CIR-DAG: [[PTR_STRIDE:%.+]] = cir.ptr_stride([[CASTED]] {{.*}}[[CONSTANT]]
// CIR-DAG: [[MINUS_ALIGN:%.+]] = cir.const #cir.int<-16>
// CIR-DAG: [[ALIGNED:%.+]] = cir.ptr_mask([[PTR_STRIDE]], [[MINUS_ALIGN]]
// CIR: [[ALIGN:%.+]] = cir.const #cir.int<16>
// CIR: [[CAST_ALIGNED:%.+]] = cir.cast(bitcast, [[ALIGNED]] : !cir.ptr<!u8i>), !cir.ptr<!cir.long_double<!cir.f80>>
// CIR: [[CAST_ALIGNED_VALUE:%.+]] = cir.load [[CAST_ALIGNED]]
// CIR: cir.store [[CAST_ALIGNED_VALUE]], [[RES]]
// CIR. cir.via.end
