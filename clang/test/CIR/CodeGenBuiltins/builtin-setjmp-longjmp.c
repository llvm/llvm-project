// RUN: %clang_cc1 -triple x86_64-unknown-linux -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -O2 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux -O2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG
void test_setjmp(void *env) {
  // CIR-LABEL: test_setjmp
  // CIR-SAME: [[ENV:%.*]]: 
  // CIR-NEXT: [[ENV_ALLOCA:%[0-9]+]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>,
  // CIR-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: [[ENV_LOAD:%[0-9]+]] = cir.load align(8) [[ENV_ALLOCA]]
  // CIR-NEXT: [[CAST:%[0-9]+]] = cir.cast bitcast [[ENV_LOAD]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: [[ZERO:%[0-9]+]] = cir.const #cir.int<0>
  // CIR-NEXT: [[FA:%[0-9]+]] = cir.frame_address([[ZERO]])
  // CIR-NEXT: cir.store align(8) [[FA]], [[CAST]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: [[SS:%[0-9]+]] = cir.stacksave
  // CIR-NEXT: [[TWO:%[0-9]+]] = cir.const #cir.int<2>
  // CIR-NEXT: [[GEP:%[0-9]+]] = cir.ptr_stride [[CAST]], [[TWO]] : (!cir.ptr<!cir.ptr<!void>>, !s32i) -> !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: cir.store align(8) [[SS]], [[GEP]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: [[SJ:%[0-9]+]] = cir.eh.setjmp [[CAST]] : (!cir.ptr<!cir.ptr<!void>>) -> !s32i


  // LLVM-LABEL: test_setjmp
  // LLVM-SAME: (ptr{{.*}}[[ENV:%.*]])
  // LLVM-NEXT: [[FA:%[0-9]+]] = {{.*}}@llvm.frameaddress.p0(i32 0) 
  // LLVM-NEXT: store ptr [[FA]], ptr [[ENV]], align 8
  // LLVM-NEXT: [[SS:%[0-9]+]] = {{.*}}@llvm.stacksave.p0() 
  // LLVM-NEXT: [[GEP:%[0-9]+]] = getelementptr i8, ptr [[ENV]], i64 16
  // LLVM-NEXT: store ptr [[SS]], ptr [[GEP]], align 8
  // LLVM-NEXT: @llvm.eh.sjlj.setjmp(ptr{{.*}}[[ENV]])
  
  // OGCG-LABEL: test_setjmp
  // OGCG-SAME: (ptr{{.*}}[[ENV:%.*]])
  // OGCG: [[FA:%.*]] = {{.*}}@llvm.frameaddress.p0(i32 0) 
  // OGCG-NEXT: store ptr [[FA]], ptr [[ENV]], align 8
  // OGCG-NEXT: [[SS:%.*]] = {{.*}}@llvm.stacksave.p0() 
  // OGCG-NEXT: [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr [[ENV]], i64 16
  // OGCG-NEXT: store ptr [[SS]], ptr [[GEP]], align 8
  // OGCG-NEXT: @llvm.eh.sjlj.setjmp(ptr{{.*}}[[ENV]])
  __builtin_setjmp(env);
}

void test_longjmp(void *env) {
  // CIR-LABEL: test_longjmp
  // CIR-SAME: [[ENV:%.*]]: 
  // CIR-NEXT: [[ENV_ALLOCA:%[0-9]+]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>,
  // CIR-NEXT: cir.store [[ENV]], [[ENV_ALLOCA]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: [[ENV_LOAD:%[0-9]+]] = cir.load align(8) [[ENV_ALLOCA]]
  // CIR-NEXT: [[CAST:%[0-9]+]] = cir.cast bitcast [[ENV_LOAD]] : !cir.ptr<!void> -> !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: cir.eh.longjmp [[CAST]] : !cir.ptr<!cir.ptr<!void>>
  // CIR-NEXT: cir.unreachable


  // LLVM-LABEL: test_longjmp
  // LLVM: @llvm.eh.sjlj.longjmp
  // LLVM-NEXT: unreachable
  
  // OGCG-LABEL: test_longjmp
  // OGCG: @llvm.eh.sjlj.longjmp
  // OGCG-NEXT: unreachable
  __builtin_longjmp(env, 1);
}
