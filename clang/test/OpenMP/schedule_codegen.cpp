// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm -fexceptions -fcxx-exceptions -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

int main() {
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for simd
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for schedule(static)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for simd schedule(static)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for schedule(static, 2)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for simd schedule(static, 2)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for schedule(auto)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for simd schedule(auto)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for schedule(runtime)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for simd schedule(runtime)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for schedule(guided)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for simd schedule(guided)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for schedule(dynamic)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for simd schedule(dynamic)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for schedule(monotonic: static)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for simd schedule(monotonic: static)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for schedule(monotonic: static, 2)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_for_static_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_for_static_fini
#pragma omp for simd schedule(monotonic: static, 2)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for schedule(monotonic: auto)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for simd schedule(monotonic: auto)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for schedule(monotonic: runtime)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for simd schedule(monotonic: runtime)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for schedule(monotonic: guided)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for simd schedule(monotonic: guided)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for schedule(monotonic: dynamic)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for simd schedule(monotonic: dynamic)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for schedule(nonmonotonic: guided)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for simd schedule(nonmonotonic: guided)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for schedule(nonmonotonic: dynamic)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK: !llvm.access.group
#pragma omp for simd schedule(nonmonotonic: dynamic)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for schedule(static) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for simd schedule(static) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for schedule(static, 2) ordered(1)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for simd schedule(static, 2) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for schedule(auto) ordered(1)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
#pragma omp for simd schedule(auto) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for schedule(runtime) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for simd schedule(runtime) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for schedule(guided) ordered(1)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for simd schedule(guided) ordered
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for schedule(dynamic) ordered(1)
  for(int i = 0; i < 10; ++i);
// CHECK: @__kmpc_dispatch_init
// CHECK-NOT: !llvm.access.group
// CHECK: @__kmpc_dispatch_next
#pragma omp for simd schedule(dynamic)
  for(int i = 0; i < 10; ++i);
  return 0;
}
