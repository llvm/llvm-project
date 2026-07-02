// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

int picked1(void) { return 0; }
int picked2(void) { return 0; }
int picked3(void);
int picked4(void);
int picked5(void) { return 0; }
int picked6(void) { return 0; }
int picked7(void) { return 0; }
int not_picked1(void) { return 1; }
int not_picked2(void) { return 2; }
int not_picked3(void);
int not_picked4(void);
int not_picked5(void);
int not_picked6(void);

#pragma omp declare variant(picked1) match(implementation={extension(match_any)}, device={kind(cpu, gpu)})
int base1(void) { return 3; }

#pragma omp declare variant(picked2) match(implementation={extension(match_none)}, device={kind(gpu, fpga)})
int base2(void) { return 4; }

#pragma omp declare variant(picked3) match(implementation={vendor(pgi), extension(match_any)}, device={kind(cpu, gpu)})
int base3(void) { return 5; }

#pragma omp declare variant(picked4) match(user={condition(0)}, implementation={extension(match_none)}, device={kind(gpu, fpga)})
int base4(void) { return 6; }

#pragma omp declare variant(picked5) match(user={condition(1)}, implementation={extension(match_all)}, device={kind(cpu)})
int base5(void) { return 7; }

#pragma omp declare variant(not_picked1) match(implementation={extension(match_any)}, device={kind(gpu, fpga)})
int base6(void) { return 0; }

#pragma omp declare variant(not_picked2) match(implementation={extension(match_none)}, device={kind(gpu, cpu)})
int base7(void) { return 0; }

#pragma omp declare variant(not_picked3) match(implementation={vendor(llvm), extension(match_any)}, device={kind(fpga, gpu)})
int base8(void) { return 0; }

#pragma omp declare variant(not_picked4) match(user={condition(1)}, implementation={extension(match_none)}, device={kind(gpu, fpga)})
int base9(void) { return 0; }

#pragma omp declare variant(not_picked5) match(user={condition(1)}, implementation={extension(match_all)}, device={kind(cpu, gpu)})
int base10(void) { return 0; }

#pragma omp declare variant(not_picked6) match(implementation={extension(match_any)})
int base11(void) { return 0; }

#pragma omp declare variant(picked6) match(implementation={extension(match_all)})
int base12(void) { return 8; }

#pragma omp declare variant(picked7) match(implementation={extension(match_none)})
int base13(void) { return 9; }

#pragma omp begin declare variant match(implementation={extension(match_any)}, device={kind(cpu, gpu)})
int overloaded1(void) { return 0; }
#pragma omp end declare variant

int overloaded2(void) { return 1; }
#pragma omp begin declare variant match(implementation={extension(match_none)}, device={kind(fpga, gpu)})
int overloaded2(void) { return 0; }
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation={extension(match_none)}, device={kind(cpu)})
NOT PARSED
#pragma omp end declare variant


int picked3(void) { return 0; }
int picked4(void) { return 0; }
int not_picked3(void) { return 10; }
int not_picked4(void) { return 11; }
int not_picked5(void) { return 12; }
int not_picked6(void) { return 13; }

int test(void) {
  // Should return 0.
  return base1() + base2() + base3() + base4() + base5() + base6() + base7() +
         base8() + base9() + base10() + base11() + base12() + base13() +
         overloaded1() + overloaded2();
}

// CHECK: call {{.*}} @{{.*}}picked1{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}picked2{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}picked3{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}picked4{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}picked5{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}base6{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}base7{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}not_picked3{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}base9{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}base10{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}base11{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}picked6{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}picked7{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}overloaded1$ompvariant$S4$s12$Pmatch_any$S2$s6$Pcpu$Pgpu
// CHECK: call {{.*}} @{{.*}}overloaded2$ompvariant$S4$s12$Pmatch_none$S2$s6$Pfpga$Pgpu
