// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -debug-info-kind=limited | FileCheck %s
// expected-no-diagnostics

int main() {
  /* int(*b)[a]; */
  /* int *(**c)[a]; */
  bool bb;
  int a;
  int b[10][10];
  int c[10][10][10];
#pragma omp target parallel firstprivate(a, b) map(tofrom          \
                                                   : c) map(tofrom \
                                                            : bb) if (a)
  {
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
#pragma omp target parallel firstprivate(a) map(tofrom         \
                                                : c, b) map(to \
                                                            : bb)
  {
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
#pragma omp target parallel map(tofrom              \
                                : a, c, b) map(from \
                                               : bb)
  {
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

// CHECK: define internal void [[DEBUG_PARALLEL:@.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* noalias{{[^,]+}}, i8 addrspace(1)* noalias{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*

// CHECK: define internal void [[NONDEBUG_WRAPPER]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* dereferenceable{{[^,]+}}, i8* dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: call void [[DEBUG_PARALLEL]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]]* dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* dereferenceable{{[^,]+}}, i8* dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: call void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define internal void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* noalias{{[^,]+}}, i8 addrspace(1)* noalias{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: addrspacecast [10 x [10 x i32]] addrspace(1)* %{{.+}} to [10 x [10 x i32]]*
// CHECK: call void [[NONDEBUG_WRAPPER:.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* {{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* {{[^,]+}}, i8* {{[^)]+}})

// CHECK: define internal void [[DEBUG_PARALLEL:@.+]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* noalias{{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* noalias{{[^,]+}}, i8 addrspace(1)* {{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]] addrspace(1)* %{{.+}} to [10 x [10 x [10 x i32]]]*
// CHECK: addrspacecast [10 x [10 x i32]] addrspace(1)* %{{.+}} to [10 x [10 x i32]]*

// CHECK: define internal void [[NONDEBUG_WRAPPER]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* dereferenceable{{[^,]+}}, i8* dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void [[DEBUG_PARALLEL]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]]* dereferenceable{{[^,]+}}, i64 {{[^,]+}}, [10 x [10 x i32]]* dereferenceable{{[^,]+}}, i8* dereferenceable{{[^)]+}})
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

// CHECK: define internal void @[[NONDEBUG_WRAPPER]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]]* dereferenceable{{[^,]+}}, i32* dereferenceable{{[^,]+}}, [10 x [10 x i32]]* dereferenceable{{[^,]+}}, i8* dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast i32* %{{.+}} to i32 addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void @[[DEBUG_PARALLEL]](i32* {{[^,]+}}, i32* {{[^,]+}}, [10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 addrspace(1)* {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: define void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]]* dereferenceable{{[^,]+}}, i32* dereferenceable{{[^,]+}}, [10 x [10 x i32]]* dereferenceable{{[^,]+}}, i8* dereferenceable{{[^)]+}})
// CHECK: addrspacecast [10 x [10 x [10 x i32]]]* %{{.+}} to [10 x [10 x [10 x i32]]] addrspace(1)*
// CHECK: addrspacecast i32* %{{.+}} to i32 addrspace(1)*
// CHECK: addrspacecast [10 x [10 x i32]]* %{{.+}} to [10 x [10 x i32]] addrspace(1)*
// CHECK: call void @__omp_offloading_{{[^(]+}}([10 x [10 x [10 x i32]]] addrspace(1)* {{[^,]+}}, i32 addrspace(1)* {{[^,]+}}, [10 x [10 x i32]] addrspace(1)* {{[^,]+}}, i8 addrspace(1)* {{[^)]+}})

// CHECK: !DILocalVariable(name: ".global_tid.",
// CHECK-SAME: DIFlagArtificial
// CHECK: !DILocalVariable(name: ".bound_tid.",
// CHECK-SAME: DIFlagArtificial
// CHECK: !DILocalVariable(name: "c",
// CHECK-SAME: line: 11
// CHECK: !DILocalVariable(name: "a",
// CHECK-SAME: line: 9
// CHECK: !DILocalVariable(name: "b",
// CHECK-SAME: line: 10

// CHECK-DAG: distinct !DISubprogram(name: "[[NONDEBUG_WRAPPER]]",
// CHECK-DAG: distinct !DISubprogram(name: "[[DEBUG_PARALLEL]]",
