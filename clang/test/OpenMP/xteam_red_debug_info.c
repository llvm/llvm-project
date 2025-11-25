// RUN: %clang -g %s -fopenmp --offload-arch=gfx90a -S --offload-host-only -emit-llvm -o - | FileCheck %s

void test_xteam_red_debug_info() {
  int N = 100000;
  double c[N];
  double sum = 0.0;
  #pragma omp target teams distribute parallel for reduction(+: sum)
  for (int i=0; i<N; i++){
    sum += c[i];
  }
  sum = sum/(double)N;
}

// CHECK:       @.offload_sizes = private unnamed_addr constant [6 x i64]
// CHECK-NEXT:  @.offload_maptypes = private unnamed_addr constant [6 x i64]
// CHECK-NEXT:  @.offload_mapnames = private constant [6 x ptr]