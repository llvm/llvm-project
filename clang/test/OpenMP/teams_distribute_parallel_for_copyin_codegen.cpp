// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

int x;
#pragma omp threadprivate(x)

template <typename T>
T tmain() {
  int a[2];
#pragma omp target
#pragma omp teams distribute parallel for copyin(x)
  for (int i = 0; i < 2; ++i) {
    a[i] = x;
  }
  return T();
}

int main() {
  int a[2];
#ifdef LAMBDA
  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
    // LAMBDA: call i32 @__tgt_target(
    // LAMBDA: call void @[[LOFFL1:.+]](
    // LAMBDA:  ret
#pragma omp target
#pragma omp teams distribute parallel for copyin(x)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void @[[LOFFL1]](
    // LAMBDA: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[LOUTL1:.+]] to {{.+}})
    // LAMBDA: ret void

    // LAMBDA: define internal void @[[LOUTL1]](
    // Skip global, bound tid and loop vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    a[i] = x;

    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[LPAR_OUTL:.+]] to
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: ret void

    // LAMBDA: define internal void @[[LPAR_OUTL]]({{.+}})
    // Skip global, bound tid and loop vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: alloca i{{[0-9]+}},
    // LAMBDA: {{%.+}} = alloca [2 x i{{[0-9]+}}]*,
    // LAMBDA: [[X_ADDR:%.+]] = alloca i{{[0-9]+}}*,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,

    // LAMBDA:  [[X_REF:%.+]] = load {{.+}}, {{.+}} [[X_ADDR]],

    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA: [[X_VAL:%.+]] = load {{.+}}, {{.+}} [[X_REF]],
    // LAMBDA: store {{.+}} [[X_VAL]],
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: ret void
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      a[i] = x;
    }();
  }
  }();
  return 0;
#else
#pragma omp target
#pragma omp teams distribute parallel for copyin(x)
  for (int i = 0; i < 2; ++i) {
    a[i] = x;
  }
  return tmain<int>();
  //return 0;
#endif
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: call i32 @__tgt_target(
// CHECK: call void @[[OFFL1:.+]](i{{64|32}} %{{.+}})
// CHECK: {{%.+}} = call{{.*}} i32 @[[TMAIN_INT:.+]]()
// CHECK:  ret

// CHECK: define{{.*}} void @[[OFFL1]]({{.+}})
// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[OUTL1:.+]] to {{.+}})
// CHECK: ret void

// CHECK: define internal void @[[OUTL1]]({{.+}})
// Skip global, bound tid and loop vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void {{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL1:.+]] to
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define internal void @[[PAR_OUTL1]]({{.+}})
// Skip global, bound tid and loop vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{%.+}} = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[X_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,

// CHECK:  [[X_REF:%.+]] = load {{.+}}, {{.+}} [[X_ADDR]],
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK:  [[X_VAL:%.+]] = load {{.+}}, {{.+}} [[X_REF]],
// CHECK: store {{.+}} [[X_VAL]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define{{.*}} i{{[0-9]+}} @[[TMAIN_INT]]()
// CHECK: call i32 @__tgt_target(
// CHECK: call void @[[TOFFL1:.+]](
// CHECK:  ret

// CHECK: define {{.*}}void @[[TOFFL1]](
// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[TOUTL1:.+]] to {{.+}})
// CHECK: ret void

// CHECK: define internal void @[[TOUTL1]]({{.+}})
// Skip global, bound tid and loop vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void {{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[TPAR_OUTL1:.+]] to
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define internal void @[[TPAR_OUTL1]]({{.+}})
// Skip global, bound tid and loop vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// prev lb and ub
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},

// CHECK: {{%.+}} = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[X_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// iter variables
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},

// CHECK:  [[X_REF:%.+]] = load {{.+}}, {{.+}} [[X_ADDR]],
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK:  [[X_VAL:%.+]] = load {{.+}}, {{.+}} [[X_REF]],
// CHECK: store {{.+}} [[X_VAL]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void


#endif
