// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#ifdef CK1

template <typename T>
int tmain(T argc) {
#pragma omp target
#pragma omp teams
  argc = 0;
  return 0;
}


int main (int argc, char **argv) {
#pragma omp target
#pragma omp teams
  {
  argc = 0;
  }
  return tmain(argv);
}

// CK1: [[MEM_TY:%.+]] = type { [128 x i8] }
// CK1-DAG: [[SHARED_GLOBAL_RD:@.+]] = common addrspace(3) global [[MEM_TY]] zeroinitializer
// CK1-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* null
// CK1-DAG: [[KERNEL_SIZE1:@.+]] = internal unnamed_addr constant i{{64|32}} 4
// CK1-DAG: [[KERNEL_SIZE2:@.+]] = internal unnamed_addr constant i{{64|32}} {{8|4}}
// CK1-DAG: [[KERNEL_SHARED1:@.+]] = internal unnamed_addr constant i16 1
// CK1-DAG: [[KERNEL_SHARED2:@.+]] = internal unnamed_addr constant i16 1

// only nvptx side: do not outline teams region and do not call fork_teams
// CK1:  define {{.*}}void @{{[^,]+}}(i{{[0-9]+}} [[ARGC:%.+]])
// CK1:  {{.+}} = alloca i{{[0-9]+}}*,
// CK1:  {{.+}} = alloca i{{[0-9]+}}*,
// CK1:  [[ARGCADDR_PTR:%.+]] = alloca i{{[0-9]+}}*,
// CK1:  [[ARGCADDR:%.+]] = alloca i{{[0-9]+}},
// CK1:  store {{.+}} 0, {{.+}},
// CK1:  store i{{[0-9]+}} [[ARGC]], i{{[0-9]+}}* [[ARGCADDR]],
// CK1-64:  [[CONV:%.+]] = bitcast i{{[0-9]+}}* [[ARGCADDR]] to i{{[0-9]+}}*
// CK1: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED1]],
// CK1: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE1]],
// CK1: call void @__kmpc_get_team_static_memory(i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[IS_SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CK1: [[KERNEL_RD:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CK1: [[GLOBALSTACK:%.+]] = getelementptr inbounds i8, i8* [[KERNEL_RD]], i{{64|32}} 0
// CK1-64:  [[ARG:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CONV]]
// CK1-32:  [[ARG:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ARGCADDR]]
// CK1:  [[ARGCADDR:%.+]] = getelementptr inbounds %struct.{{.*}}, %struct.{{.*}}* %{{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CK1:  store i{{[0-9]+}} [[ARG]], i{{[0-9]+}}* [[ARGCADDR]],
// CK1:  store i{{[0-9]+}}* [[ARGCADDR]], i{{[0-9]+}}** [[ARGCADDR_PTR]],
// CK1:  [[ARGCADDR_PTR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[ARGCADDR_PTR]],
// CK1:  store i{{[0-9]+}} 0, i{{[0-9]+}}* [[ARGCADDR_PTR_REF]],
// CK1-NOT: call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(
// CK1:  ret void
// CK1-NEXT: }

// target region in template
// CK1: define {{.*}}void @{{[^,]+}}(i{{.+}}** [[ARGC:%.+]])
// CK1: [[ARGCADDR_PTR:%.+]] = alloca i{{.+}}***,
// CK1: [[ARGCADDR:%.+]] = alloca i{{.+}}**,
// CK1: store i{{.+}}** [[ARGC]], i{{.+}}*** [[ARGCADDR]]
// CK1: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED2]],
// CK1: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE2]],
// CK1: call void @__kmpc_get_team_static_memory(i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[IS_SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CK1: [[KERNEL_RD:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CK1: [[GLOBALSTACK:%.+]] = getelementptr inbounds i8, i8* [[KERNEL_RD]], i{{64|32}} 0
// CK1: [[ARG:%.+]] = load i{{[0-9]+}}**, i{{[0-9]+}}*** [[ARGCADDR]]
// CK1: [[ARGCADDR:%.+]] = getelementptr inbounds %struct.{{.*}}, %struct.{{.*}}* %{{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CK1: store i{{[0-9]+}}** [[ARG]], i{{[0-9]+}}*** [[ARGCADDR]],
// CK1: store i8*** [[ARGCADDR]], i8**** [[ARGCADDR_PTR]],
// CK1: [[ARGCADDR_PTR_REF:%.+]] = load i{{.+}}**, i{{.+}}*** [[ARGCADDR_PTR]],
// CK1: store i{{[0-9]+}}** null, i{{[0-9]+}}*** [[ARGCADDR_PTR_REF]],
// CK1-NOT: call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(
// CK1:  ret void
// CK1-NEXT: }


#endif // CK1

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
// expected-no-diagnostics
#ifdef CK2

template <typename T>
int tmain(T argc) {
  int a = 10;
  int b = 5;
#pragma omp target
#pragma omp teams num_teams(a) thread_limit(b)
  {
  argc = 0;
  }
  return 0;
}

int main (int argc, char **argv) {
  int a = 20;
  int b = 5;
#pragma omp target
#pragma omp teams num_teams(a) thread_limit(b)
  {
  argc = 0;
  }
  return tmain(argv);
}

// CK2: [[MEM_TY:%.+]] = type { [128 x i8] }
// CK2-DAG: [[SHARED_GLOBAL_RD:@.+]] = common addrspace(3) global [[MEM_TY]] zeroinitializer
// CK2-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* null
// CK2-DAG: [[KERNEL_SIZE1:@.+]] = internal unnamed_addr constant i{{64|32}} 4
// CK2-DAG: [[KERNEL_SIZE2:@.+]] = internal unnamed_addr constant i{{64|32}} {{8|4}}
// CK2-DAG: [[KERNEL_SHARED1:@.+]] = internal unnamed_addr constant i16 1
// CK2-DAG: [[KERNEL_SHARED2:@.+]] = internal unnamed_addr constant i16 1

// CK2: define {{.*}}void @{{[^,]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}} [[B_IN:%.+]], i{{[0-9]+}} [[ARGC_IN:.+]])
// CK2: {{.}} = alloca i{{[0-9]+}}*,
// CK2: {{.}} = alloca i{{[0-9]+}}*,
// CK2: [[ARGCADDR_PTR:%.+]] = alloca i{{[0-9]+}}*,
// CK2: [[AADDR:%.+]] = alloca i{{[0-9]+}},
// CK2: [[BADDR:%.+]] = alloca i{{[0-9]+}},
// CK2: [[ARGCADDR:%.+]] = alloca i{{[0-9]+}},
// CK2: store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[AADDR]],
// CK2: store i{{[0-9]+}} [[B_IN]], i{{[0-9]+}}* [[BADDR]],
// CK2: store i{{[0-9]+}} [[ARGC_IN]], i{{[0-9]+}}* [[ARGCADDR]],
// CK2-64: [[ACONV:%.+]] = bitcast i64* [[AADDR]] to i32*
// CK2-64: [[BCONV:%.+]] = bitcast i64* [[BADDR]] to i32*
// CK2-64: [[CONV:%.+]] = bitcast i64* [[ARGCADDR]] to i32*
// CK2: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED1]],
// CK2: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE1]],
// CK2: call void @__kmpc_get_team_static_memory(i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[IS_SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CK2: [[KERNEL_RD:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CK2: [[GLOBALSTACK:%.+]] = getelementptr inbounds i8, i8* [[KERNEL_RD]], i{{64|32}} 0
// CK2-64:  [[ARG:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[CONV]]
// CK2-32:  [[ARG:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ARGCADDR]]
// CK2:  [[ARGCADDR:%.+]] = getelementptr inbounds %struct.{{.*}}, %struct.{{.*}}* %{{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CK2:  store i{{[0-9]+}} [[ARG]], i{{[0-9]+}}* [[ARGCADDR]],
// CK2:  {{%.+}} = call i32 @__kmpc_global_thread_num(
// CK2:  store i{{[0-9]+}}* [[ARGCADDR]], i{{[0-9]+}}** [[ARGCADDR_PTR]],
// CK2:  [[ARGCADDR_PTR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[ARGCADDR_PTR]],
// CK2: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[ARGCADDR_PTR_REF]],
// CK2-NOT:  {{.+}} = call i32 @__kmpc_push_num_teams(
// CK2-NOT:  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(
// CK2: ret

// CK2: define {{.*}}void @{{[^,]+}}(i{{[0-9]+}} [[A_IN:%.+]], i{{[0-9]+}} [[BP:%.+]], i{{[0-9]+}}** [[ARGC:%.+]])
// CK2: [[ARGCADDR_PTR:%.+]] = alloca i{{[0-9]+}}***,
// CK2: [[AADDR:%.+]] = alloca i{{[0-9]+}},
// CK2: [[BADDR:%.+]] = alloca i{{[0-9]+}},
// CK2: [[ARGCADDR:%.+]] = alloca i{{[0-9]+}}**,
// CK2: store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[AADDR]],
// CK2: store i{{[0-9]+}} [[B_IN]], i{{[0-9]+}}* [[BADDR]],
// CK2: store i{{[0-9]+}}** [[ARGC]], i{{[0-9]+}}*** [[ARGCADDR]],
// CK2: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED2]],
// CK2: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE2]],
// CK2: call void @__kmpc_get_team_static_memory(i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[IS_SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CK2: [[KERNEL_RD:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CK2: [[GLOBALSTACK:%.+]] = getelementptr inbounds i8, i8* [[KERNEL_RD]], i{{64|32}} 0
// CK2: [[ARG:%.+]] = load i{{[0-9]+}}**, i{{[0-9]+}}*** [[ARGCADDR]]
// CK2: [[ARGCADDR:%.+]] = getelementptr inbounds %struct.{{.*}}, %struct.{{.*}}* %{{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CK2: store i{{[0-9]+}}** [[ARG]], i{{[0-9]+}}*** [[ARGCADDR]],
// CK2: {{%.+}} = call i32 @__kmpc_global_thread_num(
// CK2: store i{{[0-9]+}}*** [[ARGCADDR]], i{{[0-9]+}}**** [[ARGCADDR_PTR]],
// CK2: [[ARGCADDR_PTR_REF:%.+]] = load i{{[0-9]+}}***, i{{[0-9]+}}**** [[ARGCADDR_PTR]],
// CK2: store i{{[0-9]+}}** null, i{{[0-9]+}}*** [[ARGCADDR_PTR_REF]],
// CK2-NOT: {{.+}} = call i32 @__kmpc_push_num_teams(
// CK2-NOT: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(
// CK2:  ret void

#endif // CK2
#endif
