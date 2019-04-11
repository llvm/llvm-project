// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <typename T>
T tmain() {
  T t_var = T();
  T vec[] = {1, 2};
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(+: t_var)
  for (int i = 0; i < 2; ++i) {
    t_var += (T) i;
  }
  return T();
}

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
    // LAMBDA: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 1)
    // LAMBDA: call void @[[LOFFL1:.+]](
    // LAMBDA:  ret
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(+: sivar)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void @[[LOFFL1]](i{{64|32}} [[SIVAR_ARG:%.+]])
    // LAMBDA: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
    // LAMBDA: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
    // LAMBDA: [[SIVAR_CONV:%.+]] = bitcast{{.+}} [[SIVAR_ADDR]] to
    // LAMBDA: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[LOUTL1:.+]] to {{.+}}, {{.+}} [[SIVAR_CONV]])
    // LAMBDA: ret void

    // LAMBDA: define internal void @[[LOUTL1]]({{.+}}, {{.+}}, {{.+}} [[SIVAR_ARG:%.+]])
    // Skip global and bound tid vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: [[SIVAR_ADDR:%.+]] = alloca i{{.+}}*,
    // LAMBDA: alloca i{{.+}},
    // LAMBDA: alloca i{{.+}},
    // LAMBDA: alloca i{{.+}},
    // LAMBDA: alloca i{{.+}},
    // LAMBDA: alloca i{{.+}},
    // LAMBDA: alloca i{{.+}},
    // LAMBDA: [[SIVAR_PRIV:%.+]] = alloca i{{.+}},
    // LAMBDA: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
    // LAMBDA: [[SIVAR_REF:%.+]] = load{{.+}}, {{.+}} [[SIVAR_ADDR]]
    // LAMBDA: store{{.+}} 0, {{.+}} [[SIVAR_PRIV]],

    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA: store{{.+}}, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](
    // LAMBDA: call void @__kmpc_for_static_fini(
    // LAMBDA: [[LAST_ITER:%.+]] = load i32, i32* %
    // LAMBDA: [[IS_LAST:%.+]] = icmp ne i32 [[LAST_ITER]], 0
    // LAMBDA: br i1 [[IS_LAST]], label %[[THEN:.+]], label %[[DONE:.+]]
    // LAMBDA: [[THEN]]
    // LAMBDA: store i32 2, i32* %
    // LAMBDA: br label %[[DONE]]
    // LAMBDA: [[DONE]]
    // LAMBDA: [[SIVAR_ORIG_VAL:%.+]] = load i32, i32* [[SIVAR_REF]],
    // LAMBDA: [[SIVAR_PRIV_VAL:%.+]] = load i32, i32* [[SIVAR_PRIV]],
    // LAMBDA: [[ADD:%.+]] = add nsw i32 [[SIVAR_ORIG_VAL]], [[SIVAR_PRIV_VAL]]
    // LAMBDA: store i32 [[ADD]], i32* [[SIVAR_REF]],
    // LAMBDA: ret void

    sivar += i;

    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],

      sivar += 4;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]

      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIVAR_REF]]
      // LAMBDA: [[SIVAR_INC:%.+]] = add{{.+}} [[SIVAR_VAL]], 4
      // LAMBDA: store i{{[0-9]+}} [[SIVAR_INC]], i{{[0-9]+}}* [[SIVAR_REF]]
    }();
  }
  }();
  return 0;
#else
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(+: sivar)
  for (int i = 0; i < 2; ++i) {
    sivar += i;
  }
  return tmain<int>();
#endif
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 1)
// CHECK: call void @[[OFFL1:.+]](i{{64|32}} %{{.+}})
// CHECK: {{%.+}} = call{{.*}} i32 @[[TMAIN_INT:.+]]()
// CHECK:  ret

// CHECK: define{{.*}} void @[[OFFL1]](i{{64|32}} [[SIVAR_ARG:%.+]])
// CHECK: [[SIVAR_ADDR:%.+]] = alloca i{{.+}},
// CHECK: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
// CHECK-64: [[SIVAR_CONV:%.+]] = bitcast{{.+}} [[SIVAR_ADDR]] to
// CHECK-64: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}} [[SIVAR_CONV]])
// CHECK-32: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}} [[SIVAR_ADDR]])
// CHECK: ret void

// CHECK: define internal void @[[OUTL1]]({{.+}}, {{.+}}, {{.+}} [[SIVAR_ARG:%.+]])
// Skip global and bound tid vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: [[SIVAR_ADDR:%.+]] = alloca i{{.+}}*,
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{.+}},
// CHECK: store{{.+}} [[SIVAR_ARG]], {{.+}} [[SIVAR_ADDR]],
// CHECK: [[SIVAR_REF:%.+]] = load{{.+}}, {{.+}} [[SIVAR_ADDR]]
// CHECK: store{{.+}} 0, {{.+}} [[SIVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: store{{.+}}, {{.+}} [[SIVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[LAST_ITER:%.+]] = load i32, i32* %
// CHECK: [[IS_LAST:%.+]] = icmp ne i32 [[LAST_ITER]], 0
// CHECK: br i1 [[IS_LAST]], label %[[THEN:.+]], label %[[DONE:.+]]
// CHECK: [[THEN]]
// CHECK: store i32 2, i32* %
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]
// CHECK: [[SIVAR_ORIG_VAL:%.+]] = load i32, i32* [[SIVAR_REF]],
// CHECK: [[SIVAR_PRIV_VAL:%.+]] = load i32, i32* [[SIVAR_PRIV]],
// CHECK: [[ADD:%.+]] = add nsw i32 [[SIVAR_ORIG_VAL]], [[SIVAR_PRIV_VAL]]
// CHECK: store i32 [[ADD]], i32* [[SIVAR_REF]],
// CHECK: ret void

// CHECK: define{{.*}} i{{[0-9]+}} @[[TMAIN_INT]]()
// CHECK: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1,
// CHECK: call void @[[TOFFL1:.+]]({{.+}})
// CHECK:  ret

// CHECK: define{{.*}} void @[[TOFFL1]](i{{64|32}} [[TVAR_ARG:%.+]])
// CHECK: [[TVAR_ADDR:%.+]] = alloca i{{.+}},
// CHECK: store{{.+}} [[TVAR_ARG]], {{.+}} [[TVAR_ADDR]],
// CHECK-64: [[TVAR_CONV:%.+]] = bitcast{{.+}} [[TVAR_ADDR]] to
// CHECK-64: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[TOUTL1:.+]] to {{.+}}, {{.+}} [[TVAR_CONV]])
// CHECK-32: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[TOUTL1:.+]] to {{.+}}, {{.+}} [[TVAR_ADDR]])
// CHECK: ret void

// CHECK: define internal void @[[TOUTL1]]({{.+}}, {{.+}}, {{.+}} [[TVAR_ARG:%.+]])
// Skip global and bound tid vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: [[TVAR_ADDR:%.+]] = alloca i{{.+}}*,
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: alloca i{{.+}},
// CHECK: [[TVAR_PRIV:%.+]] = alloca i{{.+}},
// CHECK: store{{.+}} [[TVAR_ARG]], {{.+}} [[TVAR_ADDR]],
// CHECK: [[TVAR_REF:%.+]] = load{{.+}}, {{.+}} [[TVAR_ADDR]]
// CHECK: store{{.+}} 0, {{.+}} [[TVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: store{{.+}}, {{.+}} [[TVAR_PRIV]],
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: [[LAST_ITER:%.+]] = load i32, i32* %
// CHECK: [[IS_LAST:%.+]] = icmp ne i32 [[LAST_ITER]], 0
// CHECK: br i1 [[IS_LAST]], label %[[THEN:.+]], label %[[DONE:.+]]
// CHECK: [[THEN]]
// CHECK: store i32 2, i32* %
// CHECK: br label %[[DONE]]
// CHECK: [[DONE]]
// CHECK: [[TVAR_ORIG_VAL:%.+]] = load i32, i32* [[TVAR_REF]],
// CHECK: [[TVAR_PRIV_VAL:%.+]] = load i32, i32* [[TVAR_PRIV]],
// CHECK: [[ADD:%.+]] = add nsw i32 [[TVAR_ORIG_VAL]], [[TVAR_PRIV_VAL]]
// CHECK: store i32 [[ADD]], i32* [[TVAR_REF]],
// CHECK: ret void

// CHECK: !{!"llvm.loop.vectorize.enable", i1 true}
#endif
