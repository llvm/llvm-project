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
#pragma omp target teams distribute parallel for map(tofrom:a) if(target:argc) schedule(static, a)
  for (int i= 0; i < argc; ++i)
    a = foo(&i) + foo(&a) + foo(&argc);
  return 0;
}

// CHECK: @__omp_offloading_{{.*}}_main_l16_exec_mode = weak constant i8 0

// CHECK: define weak void @__omp_offloading_{{.*}}_main_l16(i{{64|32}} %{{[^,].*}}, i32* dereferenceable{{[^,]*}}, i{{64|32}} %{{[^,)]*}})
// CHECK: call void @__kmpc_spmd_kernel_init(
// CHECK: [[TID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @
// CHECK: call void @__kmpc_for_static_init_4(

// CHECK: call void [[PARALLEL:@.+]](i32* %{{.*}}, i32* %{{.+}}, i{{64|32}} %{{.+}}, i{{64|32}} %{{.*}}, i{{64|32}} %{{.*}}, i32* %{{.*}})
// CHECK: br label %


// CHECK: call void @__kmpc_for_static_fini(%struct.ident_t* @

// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)

// CHECK: define internal void [[PARALLEL]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i{{64|32}} %{{.+}}, i{{64|32}} %{{.+}}, i{{64|32}} [[ARGC:%.+]], i32* dereferenceable{{.*}})
// CHECK-NOT: call i8* @__kmpc_data_sharing_push_stack(
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[ARGC_ADDR:%.+]] = alloca i{{32|64}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[I:%.+]] = alloca i32,
// CHECK-32: store i32 [[ARGC]], i32* [[ARGC_ADDR]],
// CHECK-64: store i{{64|32}} [[ARGC]], i{{64|32}}* [[ARGC_ADDR]],
// CHECK-64: [[ARGC:%.+]] = bitcast i64* [[ARGC_ADDR]] to i32*

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call i32 [[FOO:@.+foo.+]](i32* [[I]])
// CHECK: call i32 [[FOO]](i32* %{{.+}})
// CHECK-32: call i32 [[FOO]](i32* [[ARGC_ADDR]])
// CHECK-64: call i32 [[FOO]](i32* [[ARGC]])
// CHECK: call void @__kmpc_for_static_fini(

// CHECK-NOT: call void @__kmpc_data_sharing_pop_stack(

#endif
