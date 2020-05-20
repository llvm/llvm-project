// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-cuda-mode -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited | FileCheck %s
// expected-no-diagnostics

int main() {
  /* int(*b)[a]; */
  /* int *(**c)[a]; */
  bool bb;
  int a;
  int b[10][10];
  int c[10][10][10];
#pragma omp target parallel for firstprivate(a, b) map(tofrom          \
                                                       : c) map(tofrom \
                                                                : bb) if (a)
  for (int i = 0; i < 10; ++i) {
    int &f = c[1][1][1];
    int &g = a;
    int &h = b[1][1];
    int d = 15;
    a = 5;
    b[0][a] = 10;
    c[0][0][a] = 11;
    b[0][a] = c[0][0][a];
    bb |= b[0][a];
  }
#pragma omp target parallel for firstprivate(a) map(tofrom         \
                                                    : c, b) map(to \
                                                                : bb)
  for (int i = 0; i < 10; ++i) {
    int &f = c[1][1][1];
    int &g = a;
    int &h = b[1][1];
    int d = 15;
    a = 5;
    b[0][a] = 10;
    c[0][0][a] = 11;
    b[0][a] = c[0][0][a];
    d = bb;
  }
#pragma omp target parallel for map(tofrom              \
                                    : a, c, b) map(from \
                                                   : bb)
  for (int i = 0; i < 10; ++i) {
    int &f = c[1][1][1];
    int &g = a;
    int &h = b[1][1];
    int d = 15;
    a = 5;
    b[0][a] = 10;
    c[0][0][a] = 11;
    b[0][a] = c[0][0][a];
    bb = b[0][a];
  }
  return 0;
}

// CHECK: define internal void @__omp_offloading{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8 addrspace(1)* noalias{{[^,]+}}, i1 {{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: call void [[NONDEBUG_WRAPPER:.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* {{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8* {{[^)]+}})

// CHECK: define internal void [[DEBUG_PARALLEL:.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* noalias{{[^,]+}}, i8 addrspace(1)* noalias{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*

// CHECK: define internal void [[NONDEBUG_WRAPPER]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i8* nonnull align {{[0-9]+}} dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: call void [[DEBUG_PARALLEL]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define weak void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i8* nonnull align {{[0-9]+}} dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: call void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define internal void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* noalias{{[^,]+}}, i8 addrspace(1)* noalias{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: addrspacecast [10 x [10 x i32]] addrspace(1)* %{{.+}} to [10 x [10 x i32]]*
// CHECK: call void [[NONDEBUG_WRAPPER:.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* {{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8* {{[^)]+}})

// CHECK: define internal void [[DEBUG_PARALLEL:@.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* noalias{{[^,]+}}, i8 addrspace(1)* {{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: addrspacecast [10 x [10 x i32]] addrspace(1)* %{{.+}} to [10 x [10 x i32]]*

// CHECK: define internal void [[NONDEBUG_WRAPPER]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i8* nonnull align {{[0-9]+}} dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void [[DEBUG_PARALLEL]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define weak void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i8* nonnull align {{[0-9]+}} dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define internal void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 addrspace(1)* noalias{{[^,]+}}, [10 x [10 x i32]] addrspace(1)* noalias{{[^,]+}}, i8 addrspace(1)* noalias{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: addrspacecast i32 addrspace(1)* %{{.+}} to i32*
// CHECK: addrspacecast [10 x [10 x i32]] addrspace(1)* %{{.+}} to [10 x [10 x i32]]*
// CHECK: call void @[[NONDEBUG_WRAPPER:.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8* {{[^)]+}})

// CHECK: define internal void @[[DEBUG_PARALLEL:.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 addrspace(1)* noalias{{[^,]+}}, [10 x [10 x i32]] addrspace(1)* noalias{{[^,]+}}, i8 addrspace(1)* noalias{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: addrspacecast i32 addrspace(1)* %{{.+}} to i32*
// CHECK: addrspacecast [10 x [10 x i32]] addrspace(1)* %{{.+}} to [10 x [10 x i32]]*

// CHECK: define internal void @[[NONDEBUG_WRAPPER]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i32* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i8* nonnull align {{[0-9]+}} dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast i32* %{{.+}} to i32 addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void @[[DEBUG_PARALLEL]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 addrspace(1)* {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define weak void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i32* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{[^,]+}}, i8* nonnull align {{[0-9]+}} dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast i32* %{{.+}} to i32 addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 addrspace(1)* {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK-DAG: distinct !DISubprogram(name: "[[NONDEBUG_WRAPPER]]",
// CHECK-DAG: distinct !DISubprogram(name: "[[DEBUG_PARALLEL]]",
