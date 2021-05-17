// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -debug-info-kind=limited -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <typename tx, typename ty>
struct TT {
  tx X;
  ty Y;
};

// TCHECK-DAG:  [[TT:%.+]] = type { i64, i8 }
// TCHECK-DAG:  [[TTII:%.+]] = type { i32, i32 }
// TCHECK-DAG:  [[S1:%.+]] = type { double }

// TCHECK: @__omp_offloading_firstprivate__{{.+}}_e_l27 = internal addrspace(4) global [[TTII]] zeroinitializer
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

  // TCHECK:  define {{.*}}void @__omp_offloading_{{.+}}([10 x float] addrspace(1)* noalias [[B_IN:%.+]], i{{[0-9]+}} [[A_IN:%.+]], [[TTII]]* noalias [[E_IN:%.+]])
  // TCHECK-NOT: alloca [[TTII]],
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: alloca [[TTII]],
  // TCHECK-NOT: alloca i{{[0-9]+}},
  // TCHECK-64:  call void @llvm.dbg.declare(metadata [10 x float] addrspace(1)** %{{.+}}, metadata !{{[0-9]+}}, metadata !DIExpression())
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
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
  // TCHECK:  define {{.*}}void @__omp_offloading_{{.+}}(i{{[0-9]+}}{{.*}} [[A2_IN:%.+]], [10 x float]*{{.*}} [[B_IN:%.+]], [5 x [10 x double]]*{{.*}} [[C_IN:%.+]], [[TT]]*{{.*}} [[D_IN:%.+]])
  // TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B_ADDR:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[C_ADDR:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[D_ADDR:%.+]] = alloca [[TT]]*,
  // TCHECK-NOT: alloca i{{[0-9]+}},
  // TCHECK:  [[B_PRIV:%.+]] = alloca [10 x float],
  // TCHECK:  [[C_PRIV:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[D_PRIV:%.+]] = alloca [[TT]],
  // TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  store [10 x float]* [[B_IN]], [10 x float]** [[B_ADDR]],
  // TCHECK:  store [5 x [10 x double]]* [[C_IN]], [5 x [10 x double]]** [[C_ADDR]],
  // TCHECK:  store [[TT]]* [[D_IN]], [[TT]]** [[D_ADDR]],
  // TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x float]*, [10 x float]** [[B_ADDR]],
  // TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x float]*, [10 x float]** %
  // TCHECK:  [[C_ADDR_REF:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[C_ADDR]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** %
  // TCHECK:  [[D_ADDR_REF:%.+]] = load [[TT]]*, [[TT]]** [[D_ADDR]],
  // TCHECK:  [[D_ADDR_REF:%.+]] = load [[TT]]*, [[TT]]** %

  // firstprivate(aa): a_priv = a_in

  //  firstprivate(b): memcpy(b_priv,b_in)
  // TCHECK:  [[B_PRIV_BCAST:%.+]] = bitcast [10 x float]* [[B_PRIV]] to i8*
  // TCHECK:  [[B_ADDR_REF_BCAST:%.+]] = bitcast [10 x float]* [[B_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[B_PRIV_BCAST]], i8* align {{[0-9]+}} [[B_ADDR_REF_BCAST]], {{.+}})

  // firstprivate(c)
  // TCHECK:  [[C_PRIV_BCAST:%.+]] = bitcast [5 x [10 x double]]* [[C_PRIV]] to i8*
  // TCHECK:  [[C_IN_BCAST:%.+]] = bitcast [5 x [10 x double]]* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[C_PRIV_BCAST]], i8* align {{[0-9]+}} [[C_IN_BCAST]],{{.+}})

  // firstprivate(d)
  // TCHECK:  [[D_PRIV_BCAST:%.+]] = bitcast [[TT]]* [[D_PRIV]] to i8*
  // TCHECK:  [[D_IN_BCAST:%.+]] = bitcast [[TT]]* [[D_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[D_PRIV_BCAST]], i8* align {{[0-9]+}} [[D_IN_BCAST]],{{.+}})

  // TCHECK: load i16, i16* [[A2_ADDR]],

#pragma omp target firstprivate(ptr)
  {
    ptr[0]++;
  }

  // TCHECK:  define weak void @__omp_offloading_{{.+}}(double* [[PTR_IN:%.+]])
  // TCHECK:  [[PTR_ADDR:%.+]] = alloca double*,
  // TCHECK-NOT: alloca double*,
  // TCHECK:  store double* [[PTR_IN]], double** [[PTR_ADDR]],
  // TCHECK:  [[PTR_IN_REF:%.+]] = load double*, double** [[PTR_ADDR]],
  // TCHECK-NOT:  store double* [[PTR_IN_REF]], double** {{%.+}},

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

// TCHECK: define {{.*}}void @__omp_offloading_{{.+}}(i{{[0-9]+}}{{.*}} [[A_IN:%.+]], i{{[0-9]+}}{{.*}} [[A3_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]])
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK-NOT:  alloca i{{[0-9]+}},
// TCHECK:  [[B_PRIV:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3_IN]], i{{[0-9]+}}* [[A3_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** %

// firstprivate(a): a_priv = a_in

// firstprivate(aaa)

// TCHECK-NOT:  store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}*

// firstprivate(b)
// TCHECK:  [[B_PRIV_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_PRIV]] to i8*
// TCHECK:  [[B_IN_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[B_PRIV_BCAST]], i8* align {{[0-9]+}} [[B_IN_BCAST]],{{.+}})

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

  // TCHECK: define internal void @__omp_offloading_{{.+}}([[S1]]* [[TH:%.+]], i{{[0-9]+}} [[B_IN:%.+]])
  // TCHECK:  [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK:  [[B_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: alloca i{{[0-9]+}},

  // TCHECK:  store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[B_IN]], i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],
  // TCHECK-64:  [[B_ADDR_CONV:%.+]] = bitcast i{{[0-9]+}}* [[B_ADDR]] to i{{[0-9]+}}*

  // firstprivate(b)
  // TCHECK-NOT:  store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}*

  // TCHECK: ret void
};

int bar(int n, double *ptr) {
  int a = 0;
  a += foo(n, ptr);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n);

  return a;
}

// template

// TCHECK: define internal void @__omp_offloading_{{.+}}(i{{[0-9]+}} [[A_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]])
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK-NOT: alloca i{{[0-9]+}},
// TCHECK:  [[B_PRIV:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** %

// firstprivate(a)
// TCHECK-NOT:  store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}*

// firstprivate(b)
// TCHECK:  [[B_PRIV_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_PRIV]] to i8*
// TCHECK:  [[B_IN_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[B_PRIV_BCAST]], i8* align {{[0-9]+}} [[B_IN_BCAST]],{{.+}})

// TCHECK: ret void

#endif

// TCHECK-DAG: distinct !DISubprogram(linkageName: "__omp_offloading_{{.+}}_worker",
// TCHECK-DAG: distinct !DISubprogram(linkageName: "__omp_offloading_{{.+}}_worker",
// TCHECK-DAG: distinct !DISubprogram(linkageName: "__omp_offloading_{{.+}}_worker",
// TCHECK-DAG: distinct !DISubprogram(linkageName: "__omp_offloading_{{.+}}_worker",
// TCHECK-DAG: distinct !DISubprogram(linkageName: "__omp_offloading_{{.+}}_worker",
