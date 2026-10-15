// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 1;
}

#pragma omp begin declare variant match(user = {condition(1)}, device = {kind(cpu)}, implementation = {vendor(llvm)})
#pragma omp begin declare variant match(device = {kind(cpu)}, implementation = {vendor(llvm, pgi), extension(match_any)})
#pragma omp begin declare variant match(device = {kind(any)}, implementation = {dynamic_allocators})
int also_after(void) {
  return 0;
}
int also_before(void) {
  return 0;
}
#pragma omp end declare variant
#pragma omp end declare variant
#pragma omp end declare variant

int also_after(void) {
  return 2;
}

int test(void) {
  // Should return 0.
  return also_after() + also_before();
}

#pragma omp begin declare variant match(device = {isa("sse")})
#pragma omp declare variant(test) match(device = {isa(sse)})
int equivalent_isa_trait(void);
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {isa("sse")})
#pragma omp declare variant(test) match(device = {isa("sse2")})
int non_equivalent_isa_trait(void);
#pragma omp end declare variant

// CHECK: call {{.*}} @{{.*}}also_after$ompvariant$S2$s6$Pany$Pcpu$S4$s20$s11$Pllvm$Ppgi$s12$Pmatch_any$S5$s13
// CHECK: call {{.*}} @{{.*}}also_before$ompvariant$S2$s6$Pany$Pcpu$S4$s20$s11$Pllvm$Ppgi$s12$Pmatch_any$S5$s13
