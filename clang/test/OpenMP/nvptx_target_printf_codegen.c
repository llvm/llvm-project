// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

#include <stdarg.h>

// expected-no-diagnostics
extern int printf(const char *, ...);
extern int vprintf(const char *, va_list);

// Check a simple call to printf end-to-end.
// CHECK: [[SIMPLE_PRINTF_TY:%[a-zA-Z0-9_]+]] = type { i32, i64, double }
int CheckSimple() {
    // CHECK: define {{.*}}void [[T1:@__omp_offloading_.+CheckSimple.+]]_worker()
#pragma omp target
  {
    // Entry point.
    // CHECK: define {{.*}}void [[T1]]()
    // Alloca in entry block.
    // CHECK: [[BUF:%[a-zA-Z0-9_]+]] = alloca [[SIMPLE_PRINTF_TY]]

    // CHECK: {{call|invoke}} void [[T1]]_worker()
    // CHECK: br label {{%?}}[[EXIT:.+]]
    //
    // CHECK-DAG: [[CMTID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    // CHECK-DAG: [[CMNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    // CHECK-DAG: [[CMWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
    // CHECK: [[IS_MASTER:%.+]] = icmp eq i32 [[CMTID]],
    // CHECK: br i1 [[IS_MASTER]], label {{%?}}[[MASTER:.+]], label {{%?}}[[EXIT]]
    //
    // CHECK: [[MASTER]]
    // CHECK-DAG: [[MNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    // CHECK-DAG: [[MWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
    // CHECK: [[MTMP1:%.+]] = sub i32 [[MNTH]], [[MWS]]
    // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]

    // printf in master-only basic block.
    // CHECK: [[FMT:%[0-9]+]] = load{{.*}}%fmt
    const char* fmt = "%d %lld %f";
    // CHECK: [[PTR0:%[0-9]+]] = getelementptr inbounds [[SIMPLE_PRINTF_TY]], [[SIMPLE_PRINTF_TY]]* [[BUF]], i32 0, i32 0
    // CHECK: store i32 1, i32* [[PTR0]], align 4
    // CHECK: [[PTR1:%[0-9]+]] = getelementptr inbounds [[SIMPLE_PRINTF_TY]], [[SIMPLE_PRINTF_TY]]* [[BUF]], i32 0, i32 1
    // CHECK: store i64 2, i64* [[PTR1]], align 8
    // CHECK: [[PTR2:%[0-9]+]] = getelementptr inbounds [[SIMPLE_PRINTF_TY]], [[SIMPLE_PRINTF_TY]]* [[BUF]], i32 0, i32 2

    // CHECK: store double 3.0{{[^,]*}}, double* [[PTR2]], align 8
    // CHECK: [[BUF_CAST:%[0-9]+]] = bitcast [[SIMPLE_PRINTF_TY]]* [[BUF]] to i8*
    // CHECK: [[RET:%[0-9]+]] = call i32 @vprintf(i8* [[FMT]], i8* [[BUF_CAST]])
    printf(fmt, 1, 2ll, 3.0);
  }

  return 0;
}

void CheckNoArgs() {
    // CHECK: define {{.*}}void [[T2:@__omp_offloading_.+CheckNoArgs.+]]_worker()
#pragma omp target
  {
    // Entry point.
    // CHECK: define {{.*}}void [[T2]]()

    // CHECK: {{call|invoke}} void [[T2]]_worker()
    // CHECK: br label {{%?}}[[EXIT:.+]]
    //
    // CHECK-DAG: [[CMTID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    // CHECK-DAG: [[CMNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    // CHECK-DAG: [[CMWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
    // CHECK: [[IS_MASTER:%.+]] = icmp eq i32 [[CMTID]],
    // CHECK: br i1 [[IS_MASTER]], label {{%?}}[[MASTER:.+]], label {{%?}}[[EXIT]]
    //
    // CHECK: [[MASTER]]
    // CHECK-DAG: [[MNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    // CHECK-DAG: [[MWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
    // CHECK: [[MTMP1:%.+]] = sub i32 [[MNTH]], [[MWS]]
    // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]

    // printf in master-only basic block.
    // CHECK: call i32 @vprintf({{.*}}, i8* null){{$}}
    printf("hello, world!");
  }
}

// Check that printf's alloca happens in the entry block, not inside the if
// statement.
int foo;
void CheckAllocaIsInEntryBlock() {
    // CHECK: define {{.*}}void [[T3:@__omp_offloading_.+CheckAllocaIsInEntryBlock.+]]_worker()
#pragma omp target
  {
    // Entry point.
    // CHECK: define {{.*}}void [[T3]](
    // Alloca in entry block.
    // CHECK: alloca %printf_args

    // CHECK: {{call|invoke}} void [[T3]]_worker()
    // CHECK: br label {{%?}}[[EXIT:.+]]
    //
    // CHECK-DAG: [[CMTID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    // CHECK-DAG: [[CMNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    // CHECK-DAG: [[CMWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
    // CHECK: [[IS_MASTER:%.+]] = icmp eq i32 [[CMTID]],
    // CHECK: br i1 [[IS_MASTER]], label {{%?}}[[MASTER:.+]], label {{%?}}[[EXIT]]
    //
    // CHECK: [[MASTER]]
    // CHECK-DAG: [[MNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    // CHECK-DAG: [[MWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
    // CHECK: [[MTMP1:%.+]] = sub i32 [[MNTH]], [[MWS]]
    // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]

    if (foo) {
      printf("%d", 42);
    }
  }
}
