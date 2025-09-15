// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <typename tx, typename ty>
struct TT {
  tx X;
  ty Y;
};

// TCHECK-DAG:  [[TTII:%.+]] = type { i32, i32 }
// TCHECK-DAG:  [[TT:%.+]] = type { i64, i8 }
// TCHECK-DAG:  [[S1:%.+]] = type { double }

int foo(int n, double *ptr) {
  int a = 0;
  short aa = 0;
  float b[10];
  double c[5][10];
  TT<long long, char> d;
  const TT<int, int> e = {n, n};

#pragma omp target firstprivate(a, e) map(tofrom \
                                          : b)
  {
    b[a] = a;
    b[a] += e.X;
  }

  // TCHECK:  define {{.*}}void @__omp_offloading_{{.+}}(ptr {{[^,]+}}, ptr addrspace(1) noalias noundef [[B_IN:%.+]], i{{[0-9]+}} noundef [[A_IN:%.+]], ptr noalias noundef [[E_IN:%.+]])
  // TCHECK:  [[DYN_PTR_ADDR:%.+]] = alloca ptr
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: alloca [[TTII]],
  // TCHECK: alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], ptr [[A_ADDR]],
  // TCHECK:  ret void

#pragma omp target firstprivate(aa, b, c, d)
  {
    aa += 1;
    b[2] = 1.0;
    c[1][2] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // make sure that firstprivate variables are generated in all cases and that we use those instances for operations inside the
  // target region
  // TCHECK:  define {{.*}}void @__omp_offloading_{{.+}}(ptr {{[^,]+}}, i{{[0-9]+}}{{.*}} [[A2_IN:%.+]], ptr{{.*}} [[B_IN:%.+]], ptr{{.*}} [[C_IN:%.+]], ptr{{.*}} [[D_IN:%.+]])
  // TCHECK:  [[DYN_PTR_ADDR:%.+]] = alloca ptr
  // TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B_ADDR:%.+]] = alloca ptr,
  // TCHECK:  [[C_ADDR:%.+]] = alloca ptr,
  // TCHECK:  [[D_ADDR:%.+]] = alloca ptr,
  // TCHECK-NOT: alloca i{{[0-9]+}},
  // TCHECK:  [[B_PRIV:%.+]] = alloca [10 x float],
  // TCHECK:  [[C_PRIV:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[D_PRIV:%.+]] = alloca [[TT]],
  // TCHECK:  store i{{[0-9]+}} [[A2_IN]], ptr [[A2_ADDR]],
  // TCHECK:  store ptr [[B_IN]], ptr [[B_ADDR]],
  // TCHECK:  store ptr [[C_IN]], ptr [[C_ADDR]],
  // TCHECK:  store ptr [[D_IN]], ptr [[D_ADDR]],
  // TCHECK:  [[B_ADDR_REF:%.+]] = load ptr, ptr [[B_ADDR]],
  // TCHECK:  [[B_ADDR_REF:%.+]] = load ptr, ptr %
  // TCHECK:  [[C_ADDR_REF:%.+]] = load ptr, ptr [[C_ADDR]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load ptr, ptr %
  // TCHECK:  [[D_ADDR_REF:%.+]] = load ptr, ptr [[D_ADDR]],
  // TCHECK:  [[D_ADDR_REF:%.+]] = load ptr, ptr %

  // firstprivate(aa): a_priv = a_in

  //  firstprivate(b): memcpy(b_priv,b_in)
  // TCHECK:  call void @llvm.memcpy.{{.+}}(ptr align {{[0-9]+}} [[B_PRIV]], ptr align {{[0-9]+}} [[B_ADDR_REF]], {{.+}})

  // firstprivate(c)
  // TCHECK:  call void @llvm.memcpy.{{.+}}(ptr align {{[0-9]+}} [[C_PRIV]], ptr align {{[0-9]+}} [[C_ADDR_REF]],{{.+}})

  // firstprivate(d)
  // TCHECK:  call void @llvm.memcpy.{{.+}}(ptr align {{[0-9]+}} [[D_PRIV]], ptr align {{[0-9]+}} [[D_ADDR_REF]],{{.+}})

  // TCHECK: load i16, ptr [[A2_ADDR]],

#pragma omp target firstprivate(ptr)
  {
    ptr[0]++;
  }

  // TCHECK:  define weak_odr protected ptx_kernel void @__omp_offloading_{{.+}}(ptr {{[^,]+}}, ptr noundef [[PTR_IN:%.+]])
  // TCHECK:  [[DYN_PTR_ADDR:%.+]] = alloca ptr,
  // TCHECK:  [[PTR_ADDR:%.+]] = alloca ptr,
  // TCHECK-NOT: alloca ptr,
  // TCHECK:  store ptr [[PTR_IN]], ptr [[PTR_ADDR]],
  // TCHECK:  [[PTR_IN_REF:%.+]] = load ptr, ptr [[PTR_ADDR]],
  // TCHECK-NOT:  store ptr [[PTR_IN_REF]], ptr {{%.+}},

  return a;
}

template <typename tx>
tx ftemplate(int n) {
  tx a = 0;
  tx b[10];

#pragma omp target firstprivate(a, b)
  {
    a += 1;
    b[2] += 1;
  }

  return a;
}

static int fstatic(int n) {
  int a = 0;
  char aaa = 0;
  int b[10];

#pragma omp target firstprivate(a, aaa, b)
  {
    a += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

template <typename tx>
void fconst(const tx t) {
#pragma omp target firstprivate(t)
  { }
}

// TCHECK: define {{.*}}void @__omp_offloading_{{.+}}(ptr {{[^,]+}}, i{{[0-9]+}}{{.*}} [[A_IN:%.+]], i{{[0-9]+}}{{.*}} [[A3_IN:%.+]], ptr {{.+}} [[B_IN:%.+]])
// TCHECK:  [[DYN_PTR_ADDR:%.+]] = alloca ptr
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca ptr,
// TCHECK-NOT:  alloca i{{[0-9]+}},
// TCHECK:  [[B_PRIV:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], ptr [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3_IN]], ptr [[A3_ADDR]],
// TCHECK:  store ptr [[B_IN]], ptr [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load ptr, ptr [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load ptr, ptr %

// firstprivate(a): a_priv = a_in

// firstprivate(aaa)

// TCHECK-NOT:  store i{{[0-9]+}} %{{.+}}, ptr

// firstprivate(b)
// TCHECK:  call void @llvm.memcpy.{{.+}}(ptr align {{[0-9]+}} [[B_PRIV]], ptr align {{[0-9]+}} [[B_ADDR_REF]],{{.+}})
// TCHECK:  ret void

struct S1 {
  double a;

  int r1(int n) {
    int b = n + 1;

#pragma omp target firstprivate(b)
    {
      this->a = (double)b + 1.5;
    }

    return (int)b;
  }

  // TCHECK: define internal void @__omp_offloading_{{.+}}(ptr {{[^,]+}}, ptr noundef [[TH:%.+]], i{{[0-9]+}} noundef [[B_IN:%.+]])
  // TCHECK:  [[DYN_PTR_ADDR:%.+]] = alloca ptr
  // TCHECK:  [[TH_ADDR:%.+]] = alloca ptr,
  // TCHECK:  [[B_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: alloca i{{[0-9]+}},

  // TCHECK:  store ptr [[TH]], ptr [[TH_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[B_IN]], ptr [[B_ADDR]],
  // TCHECK:  [[TH_ADDR_REF:%.+]] = load ptr, ptr [[TH_ADDR]],

  // firstprivate(b)
  // TCHECK-NOT:  store i{{[0-9]+}} %{{.+}}, ptr

  // TCHECK: ret void
};

int bar(int n, double *ptr) {
  int a = 0;
  a += foo(n, ptr);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n);

  fconst(TT<int, int>{0, 0});
  fconst(TT<char, char>{0, 0});

  return a;
}

// template

// TCHECK: define internal void @__omp_offloading_{{.+}}(ptr {{[^,]+}}, i{{[0-9]+}} noundef [[A_IN:%.+]], ptr{{.+}} noundef [[B_IN:%.+]])
// TCHECK:  [[DYN_PTR_ADDR:%.+]] = alloca ptr
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca ptr,
// TCHECK-NOT: alloca i{{[0-9]+}},
// TCHECK:  [[B_PRIV:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], ptr [[A_ADDR]],
// TCHECK:  store ptr [[B_IN]], ptr [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load ptr, ptr [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load ptr, ptr %

// firstprivate(a)
// TCHECK-NOT:  store i{{[0-9]+}} %{{.+}}, ptr

// firstprivate(b)
// TCHECK:  call void @llvm.memcpy.{{.+}}(ptr align {{[0-9]+}} [[B_PRIV]], ptr align {{[0-9]+}} [[B_ADDR_REF]],{{.+}})

// TCHECK: ret void

#endif
