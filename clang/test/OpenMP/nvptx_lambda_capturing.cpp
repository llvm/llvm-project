// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix HOST
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefixes=CLASS,FUN,CHECK
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefixes=CLASS,CHECK
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefixes=FUN,CHECK

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// HOST-DAG: = private unnamed_addr constant [11 x i64] [i64 4, i64 4, i64 0, i64 4, i64 40, i64 4, i64 4, i64 4, i64 8, i64 4, i64 4]
// HOST-DAG: = private unnamed_addr constant [11 x i64] [i64 547, i64 547, i64 544, i64 33, i64 673, i64 1407374883553936, i64 1407374883553936, i64 1407374883553936, i64 1407374883553936, i64 1407374883553936, i64 800]
// HOST-DAG: = private unnamed_addr constant [11 x i64] [i64 4, i64 4, i64 4, i64 0, i64 4, i64 40, i64 4, i64 4, i64 4, i64 8, i64 4]
// HOST-DAG: = private unnamed_addr constant [11 x i64] [i64 547, i64 547, i64 547, i64 544, i64 547, i64 673, i64 1688849860264592, i64 1688849860264592, i64 1688849860264592, i64 1688849860264592, i64 1688849860264592]
// HOST-DAG: = private unnamed_addr constant [3 x i64] [i64 4, i64 8, i64 8]
// HOST-DAG: = private unnamed_addr constant [3 x i64] [i64 547, i64 673, i64 562949953421968]
// HOST-DAG: = private unnamed_addr constant [3 x i64] [i64 4, i64 8, i64 8]
// HOST-DAG: = private unnamed_addr constant [3 x i64] [i64 547, i64 673, i64 562949953421968]
// CHECK-DAG: [[S:%.+]] = type { i32 }
// CHECK-DAG: [[CAP1:%.+]] = type { [[S]]* }
// CHECK-DAG: [[CAP2:%.+]] = type { i32*, i32*, i32*, i32**, i32* }

// CLASS: define internal void @__omp_offloading_{{.*}}_{{.*}}foo{{.*}}_l63_worker()
// CLASS: define weak void @__omp_offloading_{{.*}}_{{.*}}foo{{.*}}_l63([[S]]* {{%.+}}, [[CAP1]]* dereferenceable(8) {{%.+}})
// CLASS-NOT: getelementptr
// CLASS: br i1 %
// CLASS: call void @__omp_offloading_{{.*}}_{{.*}}foo{{.*}}_l63_worker()
// CLASS: br label %
// CLASS: br i1 %
// CLASS: call void @__kmpc_kernel_init(
// CLASS: call void @__kmpc_data_sharing_init_stack()
// CLASS: call void @llvm.memcpy.
// CLASS: [[L:%.+]] = load [[CAP1]]*, [[CAP1]]** [[L_ADDR:%.+]],
// CLASS: [[THIS_REF:%.+]] = getelementptr inbounds [[CAP1]], [[CAP1]]* [[L]], i32 0, i32 0
// CLASS: store [[S]]* [[S_:%.+]], [[S]]** [[THIS_REF]],
// CLASS: [[L:%.+]] = load [[CAP1]]*, [[CAP1]]** [[L_ADDR]],
// CLASS: call i32 [[LAMBDA1:@.+foo.+]]([[CAP1]]* [[L]])
// CLASS: ret void

// CLASS: define weak void @__omp_offloading_{{.+}}foo{{.+}}_l65([[S]]* %{{.+}}, [[CAP1]]* dereferenceable(8) %{{.+}})
// CLASS-NOT: getelementptr
// CLASS: call void [[PARALLEL:@.+]](i32* %{{.+}}, i32* %{{.+}}, [[S]]* %{{.+}}, [[CAP1]]* %{{.+}})
// CLASS: ret void

// CLASS: define internal void [[PARALLEL]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, [[S]]* %{{.+}}, [[CAP1]]* dereferenceable(8) %{{.+}})
// CLASS-NOT: getelementptr
// CLASS: call void @llvm.memcpy.
// CLASS: [[L:%.+]] = load [[CAP1]]*, [[CAP1]]** [[L_ADDR:%.+]],
// CLASS: [[THIS_REF:%.+]] = getelementptr inbounds [[CAP1]], [[CAP1]]* [[L]], i32 0, i32 0
// CLASS: store [[S]]* %{{.+}}, [[S]]** [[THIS_REF]],
// CLASS: [[L:%.+]] = load [[CAP1]]*, [[CAP1]]** [[L_ADDR]],
// CLASS: call i32 [[LAMBDA1]]([[CAP1]]* [[L]])
// CLASS: ret void

struct S {
  int a = 15;
  int foo() {
    auto &&L = [&]() { return a; };
#pragma omp target
    L();
#pragma omp target parallel
    L();
    return a;
  }
} s;

// FUN: define internal void @__omp_offloading_{{.+}}_main_l125_worker()
// FUN: define weak void @__omp_offloading_{{.+}}_main_l125(i32* dereferenceable(4) %{{.+}}, i32* dereferenceable(4) %{{.+}}, i32* %{{.+}}, i32* dereferenceable(4) %{{.+}}, [[CAP2]]* dereferenceable(40) %{{.+}}, i64 %{{.+}})
// FUN-NOT: getelementptr
// FUN: br i1 %
// FUN: call void @__omp_offloading_{{.*}}_{{.*}}main{{.*}}_l125_worker()
// FUN: br label %
// FUN: br i1 %
// FUN: call void @__kmpc_kernel_init(
// FUN: call void @__kmpc_data_sharing_init_stack()
// FUN: call void @llvm.memcpy.
// FUN: [[L:%.+]] = load [[CAP2]]*, [[CAP2]]** [[L_ADDR:%.+]],
// FUN: [[ARGC_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 0
// FUN: store i32* %{{.+}}, i32** [[ARGC_CAP]],
// FUN: [[B_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 1
// FUN: store i32* %{{.+}}, i32** [[B_CAP]],
// FUN: [[C_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 2
// FUN: store i32* %{{.+}}, i32** [[C_CAP]],
// FUN: [[D_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 3
// FUN: store i32** %{{.+}}, i32*** [[D_CAP]],
// FUN: [[A_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 4
// FUN: store i32* %{{.+}}, i32** [[A_CAP]],
// FUN: [[L:%.+]] = load [[CAP2]]*, [[CAP2]]** [[L_ADDR:%.+]],
// FUN: call i64 [[LAMBDA2:@.+main.+]]([[CAP2]]* [[L]])
// FUN: ret void

// FUN: define weak void @__omp_offloading_{{.+}}_main_l127(i32* dereferenceable(4) %{{.+}}, i32* dereferenceable(4) %{{.+}} i32* dereferenceable(4) %{{.+}}, i32* %{{.+}}, i32* dereferenceable(4) %{{.+}}, [[CAP2]]* dereferenceable(40) %{{.+}})
// FUN-NOT: getelementptr
// FUN: call void [[PARALLEL:@.+]](i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, i32* %{{.+}}, [[CAP2]]* %{{.+}})
// FUN: ret void

// FUN: define internal void [[PARALLEL:@.+]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}}, i32* dereferenceable(4) %{{.+}}, i32* dereferenceable(4) %{{.+}}, i32* %{{.+}}, i32* dereferenceable(4) %{{.+}}, [[CAP2]]* dereferenceable(40) %{{.+}})
// FUN-NOT: getelementptr
// FUN: call void @llvm.memcpy.
// FUN: [[L:%.+]] = load [[CAP2]]*, [[CAP2]]** [[L_ADDR]],
// FUN: [[ARGC_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 0
// FUN: store i32* %{{.+}}, i32** [[ARGC_CAP]],
// FUN: [[B_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 1
// FUN: store i32* %{{.+}}, i32** [[B_CAP]],
// FUN: [[C_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 2
// FUN: store i32* %{{.+}}, i32** [[C_CAP]],
// FUN: [[D_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 3
// FUN: store i32** %{{.+}}, i32*** [[D_CAP]],
// FUN: [[A_CAP:%.+]] = getelementptr inbounds [[CAP2]], [[CAP2]]* [[L]], i32 0, i32 4
// FUN: store i32* %{{.+}}, i32** [[A_CAP]],
// FUN: [[L:%.+]] = load [[CAP2]]*, [[CAP2]]** [[L_ADDR]],
// FUN: call i64 [[LAMBDA2]]([[CAP2]]* [[L]])
// FUN: ret void

int main(int argc, char **argv) {
  int &b = argc;
  int &&c = 1;
  int *d = &argc;
  int a;
  auto &&L = [&]() { return argc + b + c + reinterpret_cast<long int>(d) + a; };
#pragma omp target firstprivate(argc) map(to : a)
  L();
#pragma omp target parallel
  L();
  return argc + s.foo();
}

#endif // HEADER
