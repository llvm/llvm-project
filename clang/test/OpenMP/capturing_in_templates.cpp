// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-ibm-linux-gnu -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics

template <typename T1, typename T2>
struct pair {
  T1 t1;
  T2 t2;
  pair(T1 t1, T2 t2) : t1(t1), t2(t2) {}
};

template <typename T1, typename T2>
pair<T1, T2> make_pair(T1 &&t1, T2 &&t2) {
  return {t1, t2};
}

// CHECK-LABEL: @main
int main(int argc, char **argv) {
// CHECK: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @{{.+}}.region_id, %struct.__tgt_kernel_arguments* %{{.+}})
#pragma omp target
 {
    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 64; ++j) {
        auto foo = make_pair(i * i, j * j);
      }
    }
  }
  return 0;
}

// CHECK: call {{.+}} @{{.*}}make_pair
