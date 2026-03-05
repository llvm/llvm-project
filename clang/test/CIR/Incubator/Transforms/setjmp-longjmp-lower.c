// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s -check-prefix=BEFORE-LOWERING-PREPARE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s -check-prefix=AFTER-LOWERING-PREPARE
void test_setjmp(void *env) {
  // BEFORE-LOWERING-PREPARE-LABEL: test_setjmp
  // BEFORE-LOWERING-PREPARE-SAME: [[ENV:%.*]]: 
  // BEFORE-LOWERING-PREPARE-NEXT: [[ENV_ALLOCA:%[0-9]+]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>,
  // BEFORE-LOWERING-PREPARE-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // BEFORE-LOWERING-PREPARE-NEXT: [[ENV_LOAD:%[0-9]+]] = cir.load align(8) [[ENV_ALLOCA]]
  // BEFORE-LOWERING-PREPARE-NEXT: [[CAST:%[0-9]+]] = cir.cast bitcast [[ENV_LOAD]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!void>>
  // BEFORE-LOWERING-PREPARE-NEXT: [[ZERO:%[0-9]+]] = cir.const #cir.int<0>
  // BEFORE-LOWERING-PREPARE-NEXT: [[FA:%[0-9]+]] = cir.frame_address([[ZERO]])
  // BEFORE-LOWERING-PREPARE-NEXT: cir.store [[FA]], [[CAST]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // BEFORE-LOWERING-PREPARE-NEXT: [[SS:%[0-9]+]] = cir.stack_save
  // BEFORE-LOWERING-PREPARE-NEXT: [[TWO:%[0-9]+]] = cir.const #cir.int<2>
  // BEFORE-LOWERING-PREPARE-NEXT: [[GEP:%[0-9]+]] = cir.ptr_stride [[CAST]], [[TWO]] : (!cir.ptr<!cir.ptr<!void>>, !s32i) -> !cir.ptr<!cir.ptr<!void>>
  // BEFORE-LOWERING-PREPARE-NEXT: cir.store [[SS]], [[GEP]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // BEFORE-LOWERING-PREPARE-NEXT: [[SJ:%[0-9]+]] = cir.eh.setjmp builtin [[CAST]] : (!cir.ptr<!cir.ptr<!void>>) -> !s32i

  // AFTER-LOWERING-PREPARE-LABEL: test_setjmp
  // AFTER-LOWERING-PREPARE-SAME: [[ENV:%.*]]: 
  // AFTER-LOWERING-PREPARE-NEXT: [[ENV_ALLOCA:%[0-9]+]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>,
  // AFTER-LOWERING-PREPARE-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // AFTER-LOWERING-PREPARE-NEXT: [[ENV_LOAD:%[0-9]+]] = cir.load align(8) [[ENV_ALLOCA]]
  // AFTER-LOWERING-PREPARE-NEXT: [[CAST:%[0-9]+]] = cir.cast bitcast [[ENV_LOAD]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!void>>
  // AFTER-LOWERING-PREPARE-NEXT: [[ZERO:%[0-9]+]] = cir.const #cir.int<0>
  // AFTER-LOWERING-PREPARE-NEXT: [[FA:%[0-9]+]] = cir.frame_address([[ZERO]])
  // AFTER-LOWERING-PREPARE-NEXT: cir.store [[FA]], [[CAST]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // AFTER-LOWERING-PREPARE-NEXT: [[SS:%[0-9]+]] = cir.stack_save
  // AFTER-LOWERING-PREPARE-NEXT: [[TWO:%[0-9]+]] = cir.const #cir.int<2>
  // AFTER-LOWERING-PREPARE-NEXT: [[GEP:%[0-9]+]] = cir.ptr_stride [[CAST]], [[TWO]] : (!cir.ptr<!cir.ptr<!void>>, !s32i) -> !cir.ptr<!cir.ptr<!void>>
  // AFTER-LOWERING-PREPARE-NEXT: cir.store [[SS]], [[GEP]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // AFTER-LOWERING-PREPARE-NEXT: [[SJ:%[0-9]+]] = cir.eh.setjmp builtin [[CAST]] : (!cir.ptr<!cir.ptr<!void>>) -> !s32i
  __builtin_setjmp(env);
}

extern int _setjmp(void *env);
void test_setjmp2(void *env) {
  // BEFORE-LOWERING-PREPARE-LABEL: test_setjmp2
  // BEFORE-LOWERING-PREPARE-SAME: [[ENV:%.*]]:
  // BEFORE-LOWERING-PREPARE-NEXT: [[ENV_ALLOCA:%.*]] = cir.alloca
  // BEFORE-LOWERING-PREPARE-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]]
  // BEFORE-LOWERING-PREPARE-NEXT: [[ENV_LOAD:%.*]] = cir.load align(8) [[ENV_ALLOCA]]
  // BEFORE-LOWERING-PREPARE-NEXT: [[CAST:%.*]] = cir.cast bitcast [[ENV_LOAD]]
  // BEFORE-LOWERING-PREPARE-NEXT: cir.eh.setjmp [[CAST]] : (!cir.ptr<!cir.ptr<!void>>) -> !s32i

  // AFTER-LOWERING-PREPARE-LABEL: test_setjmp2
  // AFTER-LOWERING-PREPARE-SAME: [[ENV:%.*]]:
  // AFTER-LOWERING-PREPARE-NEXT: [[ENV_ALLOCA:%.*]] = cir.alloca
  // AFTER-LOWERING-PREPARE-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]]
  // AFTER-LOWERING-PREPARE-NEXT: [[ENV_LOAD:%.*]] = cir.load align(8) [[ENV_ALLOCA]]
  // AFTER-LOWERING-PREPARE-NEXT: [[CAST:%.*]] = cir.cast bitcast [[ENV_LOAD]]
  // AFTER-LOWERING-PREPARE-NEXT: cir.eh.setjmp [[CAST]] : (!cir.ptr<!cir.ptr<!void>>) -> !s32i
  _setjmp (env);
}
void test_longjmp(void *env) {
  // BEFORE-LOWERING-PREPARE-LABEL: test_longjmp
  // BEFORE-LOWERING-PREPARE-SAME: [[ENV:%.*]]:
  // BEFORE-LOWERING-PREPARE-NEXT: [[ENV_ALLOCA:%.*]] = cir.alloca
  // BEFORE-LOWERING-PREPARE-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]]
  // BEFORE-LOWERING-PREPARE-NEXT: [[ENV_LOAD:%.*]] = cir.load align(8) [[ENV_ALLOCA]]
  // BEFORE-LOWERING-PREPARE-NEXT: [[CAST:%.*]] = cir.cast bitcast [[ENV_LOAD]]
  // BEFORE-LOWERING-PREPARE-NEXT: cir.eh.longjmp [[CAST]] : !cir.ptr<!cir.ptr<!void>>
  // BEFORE-LOWERING-PREPARE-NEXT: cir.unreachable

  // AFTER-LOWERING-PREPARE-LABEL: test_longjmp
  // AFTER-LOWERING-PREPARE-SAME: [[ENV:%.*]]:
  // AFTER-LOWERING-PREPARE-NEXT: [[ENV_ALLOCA:%.*]] = cir.alloca
  // AFTER-LOWERING-PREPARE-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]]
  // AFTER-LOWERING-PREPARE-NEXT: [[ENV_LOAD:%.*]] = cir.load align(8) [[ENV_ALLOCA]]
  // AFTER-LOWERING-PREPARE-NEXT: [[CAST:%.*]] = cir.cast bitcast [[ENV_LOAD]]
  // AFTER-LOWERING-PREPARE-NEXT: cir.eh.longjmp [[CAST]] : !cir.ptr<!cir.ptr<!void>>
  // AFTER-LOWERING-PREPARE-NEXT: cir.unreachable
  __builtin_longjmp(env, 1);
}
