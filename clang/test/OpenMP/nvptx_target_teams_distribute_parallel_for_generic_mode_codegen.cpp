// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

int a;

int foo(int *a);

int main(int argc, char **argv) {
#pragma omp target teams distribute parallel for map(tofrom:a) if(parallel:argc)
  for (int i= 0; i < argc; ++i)
    a = foo(&i) + foo(&a) + foo(&argc);
  return 0;
}

// CHECK: define internal void @__omp_offloading_{{.*}}_main_l[[@LINE-6]]_worker()
// CHECK: [[TID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @
// CHECK: call void [[PARALLEL:@.+]]_wrapper(i16 0, i32 [[TID]])

// CHECK: define void @__omp_offloading_{{.*}}_main_l[[@LINE-10]](i{{64|32}} %{{[^,].*}}, i32* dereferenceable{{[^,]*}}, i{{64|32}} %{{[^,)]*}})
// CHECK: [[TID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @
// CHECK: call void @__kmpc_kernel_init(
// CHECK: call void @__kmpc_data_sharing_init_stack()
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_kernel_prepare_parallel(
// CHECK: call void @__kmpc_begin_sharing_variables(i8*** [[BUF_PTR_PTR:%[^,]+]], i{{64|32}} 4)
// CHECK: [[BUF_PTR:%.+]] = load i8**, i8*** [[BUF_PTR_PTR]],
// CHECK: [[LB:%.+]] = inttoptr i{{64|32}} [[LB_:%.*]] to i8*
// CHECK: store i8* [[LB]], i8** [[BUF_PTR]],
// CHECK: [[BUF_PTR1:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 1
// CHECK: [[UB:%.+]] = inttoptr i{{64|32}} [[UB_:%.*]] to i8*
// CHECK: store i8* [[UB]], i8** [[BUF_PTR1]],
// CHECK: [[BUF_PTR2:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 2
// CHECK: [[ARGC:%.+]] = inttoptr i{{64|32}} [[ARGC_:%.*]] to i8*
// CHECK: store i8* [[ARGC]], i8** [[BUF_PTR2]],
// CHECK: [[BUF_PTR3:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 3
// CHECK: [[A_PTR:%.+]] = bitcast i32* [[A_ADDR:%.*]] to i8*
// CHECK: store i8* [[A_PTR]], i8** [[BUF_PTR3]],
// CHECK: call void @llvm.nvvm.barrier0()
// CHECK: call void @llvm.nvvm.barrier0()
// CHECK: call void @__kmpc_end_sharing_variables()
// CHECK: br label

// CHECK: call void @__kmpc_serialized_parallel(%struct.ident_t* @
// CHECK: [[GTID_ADDR:%.*]] = load i32*, i32** %
// CHECK: call void [[PARALLEL]](i32* [[GTID_ADDR]], i32* %{{.+}}, i{{64|32}} [[LB_]], i{{64|32}} [[UB_]], i{{64|32}} [[ARGC_]], i32* [[A_ADDR]])
// CHECK: call void @__kmpc_end_serialized_parallel(%struct.ident_t* @
// CHECK: br label %


// CHECK: call void @__kmpc_for_static_fini(%struct.ident_t* @

// CHECK: call void @__kmpc_kernel_deinit(i16 1)
// CHECK: call void @llvm.nvvm.barrier0()

// CHECK: define internal void [[PARALLEL]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i{{64|32}} %{{.+}}, i{{64|32}} %{{.+}}, i{{64|32}} %{{.+}}, i32* dereferenceable{{.*}})
// CHECK: [[RES:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i{{64|32}} 8, i16 0)
// CHECK: [[GLOBALS:%.+]] = bitcast i8* [[RES]] to [[GLOBAL_TY:%.+]]*
// CHECK: [[I:%.+]] = getelementptr inbounds [[GLOBAL_TY]], [[GLOBAL_TY]]* [[GLOBALS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[ARGC_VAL:%.+]] = load i32, i32* %
// CHECK: [[ARGC:%.+]] = getelementptr inbounds [[GLOBAL_TY]], [[GLOBAL_TY]]* [[GLOBALS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store i32 [[ARGC_VAL]], i32* [[ARGC]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call i32 [[FOO:@.+foo.+]](i32* [[I]])
// CHECK: call i32 [[FOO]](i32* %{{.+}})
// CHECK: call i32 [[FOO]](i32* [[ARGC]])
// CHECK: call void @__kmpc_for_static_fini(

// CHECK: call void @__kmpc_data_sharing_pop_stack(i8* [[RES]])

// define internal void [[PARALLEL]]_wrapper(i16 zeroext, i32)
// CHECK: call void @__kmpc_get_shared_variables(i8*** [[BUF_PTR_PTR:%.+]])
// CHECK: [[BUF_PTR:%.+]] = load i8**, i8*** [[BUF_PTR_PTR]],
// CHECK: [[BUF_PTR0:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 0
// CHECK: [[LB_PTR:%.+]] = bitcast i8** [[BUF_PTR0]] to i{{64|32}}*
// CHECK: [[LB:%.+]] = load i{{64|32}}, i{{64|32}}* [[LB_PTR]],
// CHECK: [[BUF_PTR1:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 1
// CHECK: [[UB_PTR:%.+]] = bitcast i8** [[BUF_PTR1]] to i{{64|32}}*
// CHECK: [[UB:%.+]] = load i{{64|32}}, i{{64|32}}* [[UB_PTR]],
// CHECK: [[BUF_PTR2:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 2
// CHECK: [[ARGC_ADDR:%.+]] = bitcast i8** [[BUF_PTR2]] to i32*
// CHECK: [[ARGC:%.+]] = load i32, i32* [[ARGC_ADDR]],
// CHECK-64: [[ARGC_CAST:%.+]] = zext i32 [[ARGC]] to i64
// CHECK: [[BUF_PTR3:%.+]] = getelementptr inbounds i8*, i8** [[BUF_PTR]], i{{[0-9]+}} 3
// CHECK: [[A_ADDR_REF:%.+]] = bitcast i8** [[BUF_PTR3]] to i32**
// CHECK: [[A_ADDR:%.+]] = load i32*, i32** [[A_ADDR_REF]],
// CHECK-64: call void [[PARALLEL]](i32* %{{.+}}, i32* %{{.+}}, i64 [[LB]], i64 [[UB]], i64 [[ARGC_CAST]], i32* [[A_ADDR]])
// CHECK-32: call void [[PARALLEL]](i32* %{{.+}}, i32* %{{.+}}, i32 [[LB]], i32 [[UB]], i32 [[ARGC]], i32* [[A_ADDR]])
// CHECK: ret void

#endif
