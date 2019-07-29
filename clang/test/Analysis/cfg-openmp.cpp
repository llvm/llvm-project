// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG %s 2>&1 -fopenmp -fopenmp-version=45 | FileCheck %s

// CHECK-LABEL:  void xxx(int argc)
void xxx(int argc) {
// CHECK:        [B1]
// CHECK-NEXT:   1: int x;
// CHECK-NEXT:   2: int cond;
// CHECK-NEXT:   3: int fp;
// CHECK-NEXT:   4: int rd;
  int x, cond, fp, rd;
// CHECK-NEXT:   [[#ATOM:]]: x
// CHECK-NEXT:   [[#ATOM+1]]: [B1.[[#ATOM]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   [[#ATOM+2]]: argc
// CHECK-NEXT:   [[#ATOM+3]]: [B1.[[#ATOM+2]]] = [B1.[[#ATOM+1]]]
// CHECK-NEXT:   [[#ATOM+4]]: #pragma omp atomic read
// CHECK-NEXT:   [B1.[[#ATOM+3]]];
#pragma omp atomic read
  argc = x;
// CHECK-NEXT:   [[#CRIT:]]: x
// CHECK-NEXT:   [[#CRIT+1]]: [B1.[[#CRIT]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   [[#CRIT+2]]: argc
// CHECK-NEXT:   [[#CRIT+3]]: [B1.[[#CRIT+2]]] = [B1.[[#CRIT+1]]]
// CHECK-NEXT:   [[#CRIT+4]]: #pragma omp critical
// CHECK-NEXT:   [B1.[[#CRIT+3]]];
#pragma omp critical
  argc = x;
// CHECK-NEXT:  [[#DPF:]]: x
// CHECK-NEXT:  [[#DPF+1]]: [B1.[[#DPF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#DPF+2]]: argc
// CHECK-NEXT:  [[#DPF+3]]: [B1.[[#DPF+2]]] = [B1.[[#DPF+1]]]
// CHECK-NEXT:  [[#DPF+4]]: cond
// CHECK-NEXT:  [[#DPF+5]]: [B1.[[#DPF+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#DPF+6]]: [B1.[[#DPF+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#DPF+7]]: fp
// CHECK-NEXT:  [[#DPF+8]]: rd
// CHECK-NEXT:  [[#DPF+9]]: #pragma omp distribute parallel for if(parallel: cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#DPF+3]]];
#pragma omp distribute parallel for if(parallel:cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#DPFS:]]: x
// CHECK-NEXT:  [[#DPFS+1]]: [B1.[[#DPFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#DPFS+2]]: argc
// CHECK-NEXT:  [[#DPFS+3]]: [B1.[[#DPFS+2]]] = [B1.[[#DPFS+1]]]
// CHECK-NEXT:  [[#DPFS+4]]: cond
// CHECK-NEXT:  [[#DPFS+5]]: [B1.[[#DPFS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#DPFS+6]]: [B1.[[#DPFS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#DPFS+7]]: fp
// CHECK-NEXT:  [[#DPFS+8]]: rd
// CHECK-NEXT:  [[#DPFS+9]]: #pragma omp distribute parallel for simd if(cond) firstprivate(fp) reduction(-: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#DPFS+3]]];
#pragma omp distribute parallel for simd if(cond)  firstprivate(fp) reduction(-:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#DS:]]: x
// CHECK-NEXT:  [[#DS+1]]: [B1.[[#DS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#DS+2]]: argc
// CHECK-NEXT:  [[#DS+3]]: [B1.[[#DS+2]]] = [B1.[[#DS+1]]]
// CHECK-NEXT:  [[#DS+4]]: #pragma omp distribute simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#DS+3]]];
#pragma omp distribute simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#FOR:]]: x
// CHECK-NEXT:  [[#FOR+1]]: [B1.[[#FOR]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#FOR+2]]: argc
// CHECK-NEXT:  [[#FOR+3]]: [B1.[[#FOR+2]]] = [B1.[[#FOR+1]]]
// CHECK-NEXT:  [[#FOR+4]]: #pragma omp for
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#FOR+3]]];
#pragma omp for
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#FS:]]: x
// CHECK-NEXT:  [[#FS+1]]: [B1.[[#FS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#FS+2]]: argc
// CHECK-NEXT:  [[#FS+3]]: [B1.[[#FS+2]]] = [B1.[[#FS+1]]]
// CHECK-NEXT:  [[#FS+4]]: #pragma omp for simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#FS+3]]];
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#MASTER:]]: x
// CHECK-NEXT:  [[#MASTER+1]]: [B1.[[#MASTER]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#MASTER+2]]: argc
// CHECK-NEXT:  [[#MASTER+3]]: [B1.[[#MASTER+2]]] = [B1.[[#MASTER+1]]]
// CHECK-NEXT:  [[#MASTER+4]]: #pragma omp master
// CHECK-NEXT:    [B1.[[#MASTER+3]]];
#pragma omp master
  argc = x;
// CHECK-NEXT:  [[#ORD:]]: x
// CHECK-NEXT:  [[#ORD+1]]: [B1.[[#ORD]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#ORD+2]]: argc
// CHECK-NEXT:  [[#ORD+3]]: [B1.[[#ORD+2]]] = [B1.[[#ORD+1]]]
// CHECK-NEXT:  [[#ORD+4]]: #pragma omp ordered
// CHECK-NEXT:    [B1.[[#ORD+3]]];
// CHECK-NEXT:  [[#ORD+5]]: #pragma omp for ordered
// CHECK-NEXT:    for (int i = 0; i < 10; ++i) {
// CHECK-NEXT:[B1.[[#ORD+4]]]    }
#pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered
    argc = x;
  }
// CHECK-NEXT:  [[#PF:]]: x
// CHECK-NEXT:  [[#PF+1]]: [B1.[[#PF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PF+2]]: argc
// CHECK-NEXT:  [[#PF+3]]: [B1.[[#PF+2]]] = [B1.[[#PF+1]]]
// CHECK-NEXT:  [[#PF+4]]: cond
// CHECK-NEXT:  [[#PF+5]]: [B1.[[#PF+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PF+6]]: [B1.[[#PF+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#PF+7]]: fp
// CHECK-NEXT:  [[#PF+8]]: rd
// CHECK-NEXT:  [[#PF+9]]: #pragma omp parallel for if(cond) firstprivate(fp) reduction(&: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#PF+3]]];
#pragma omp parallel for if(cond) firstprivate(fp) reduction(&:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#PFS:]]: x
// CHECK-NEXT:  [[#PFS+1]]: [B1.[[#PFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PFS+2]]: argc
// CHECK-NEXT:  [[#PFS+3]]: [B1.[[#PFS+2]]] = [B1.[[#PFS+1]]]
// CHECK-NEXT:  [[#PFS+4]]: cond
// CHECK-NEXT:  [[#PFS+5]]: [B1.[[#PFS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PFS+6]]: [B1.[[#PFS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#PFS+7]]: fp
// CHECK-NEXT:  [[#PFS+8]]: rd
// CHECK-NEXT:  [[#PFS+9]]: #pragma omp parallel for simd if(cond) firstprivate(fp) reduction(|: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#PFS+3]]];
#pragma omp parallel for simd if(cond) firstprivate(fp) reduction(|:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#PAR:]]: x
// CHECK-NEXT:  [[#PAR+1]]: [B1.[[#PAR]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PAR+2]]: argc
// CHECK-NEXT:  [[#PAR+3]]: [B1.[[#PAR+2]]] = [B1.[[#PAR+1]]]
// CHECK-NEXT:  [[#PAR+4]]: cond
// CHECK-NEXT:  [[#PAR+5]]: [B1.[[#PAR+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PAR+6]]: [B1.[[#PAR+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#PAR+7]]: fp
// CHECK-NEXT:  [[#PAR+8]]: rd
// CHECK-NEXT:  [[#PAR+9]]: #pragma omp parallel if(cond) firstprivate(fp) reduction(min: rd)
// CHECK-NEXT:    [B1.[[#PAR+3]]];
#pragma omp parallel if(cond) firstprivate(fp) reduction(min:rd)
  argc = x;
// CHECK-NEXT:  [[#PSECT:]]: x
// CHECK-NEXT:  [[#PSECT+1]]: [B1.[[#PSECT]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PSECT+2]]: argc
// CHECK-NEXT:  [[#PSECT+3]]: [B1.[[#PSECT+2]]] = [B1.[[#PSECT+1]]]
// CHECK-NEXT:  [[#PSECT+4]]: cond
// CHECK-NEXT:  [[#PSECT+5]]: [B1.[[#PSECT+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#PSECT+6]]: [B1.[[#PSECT+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#PSECT+7]]: fp
// CHECK-NEXT:  [[#PSECT+8]]: rd
// CHECK-NEXT:  [[#PSECT+9]]: #pragma omp parallel sections if(cond) firstprivate(fp) reduction(&&: rd)
// CHECK-NEXT:    {
// CHECK-NEXT:        [B1.[[#PSECT+3]]];
// CHECK-NEXT:    }
#pragma omp parallel sections if(cond) firstprivate(fp) reduction(&&:rd)
  {
    argc = x;
  }
// CHECK-NEXT:  [[#SIMD:]]: x
// CHECK-NEXT:  [[#SIMD+1]]: [B1.[[#SIMD]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#SIMD+2]]: argc
// CHECK-NEXT:  [[#SIMD+3]]: [B1.[[#SIMD+2]]] = [B1.[[#SIMD+1]]]
// CHECK-NEXT:  [[#SIMD+4]]: #pragma omp simd
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#SIMD+3]]];
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#SINGLE:]]: x
// CHECK-NEXT:  [[#SINGLE+1]]: [B1.[[#SINGLE]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#SINGLE+2]]: argc
// CHECK-NEXT:  [[#SINGLE+3]]: [B1.[[#SINGLE+2]]] = [B1.[[#SINGLE+1]]]
// CHECK-NEXT:  [[#SINGLE+4]]: #pragma omp single
// CHECK-NEXT:    [B1.[[#SINGLE+3]]];
#pragma omp single
  argc = x;
// CHECK-NEXT:  [[#TARGET:]]:
// CHECK-SAME:  [B1.[[#TARGET+10]]]
// CHECK-NEXT:  [[#TARGET+1]]: [B1.[[#TARGET+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TARGET+2]]: [B1.[[#TARGET+9]]]
// CHECK-NEXT:  [[#TARGET+3]]: [B1.[[#TARGET+9]]] = [B1.[[#TARGET+1]]]
// CHECK-NEXT:  [[#TARGET+4]]: cond
// CHECK-NEXT:  [[#TARGET+5]]: [B1.[[#TARGET+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TARGET+6]]: [B1.[[#TARGET+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TARGET+7]]: fp
// CHECK-NEXT:  [[#TARGET+8]]: rd
// CHECK-NEXT:  [[#TARGET+9]]: argc
// CHECK-NEXT:  [[#TARGET+10]]: x
// CHECK-NEXT:  [[#TARGET+11]]: #pragma omp target depend(in : argc) if(cond) firstprivate(fp) reduction(-: rd)
// CHECK-NEXT:    [B1.[[#TARGET+3]]];
#pragma omp target depend(in \
                          : argc) if(cond) firstprivate(fp) reduction(-:rd)
  argc = x;
// CHECK-NEXT:  [[#TPF:]]:
// CHECK-SAME:  [B1.[[#TPF+10]]]
// CHECK-NEXT:  [[#TPF+1]]: [B1.[[#TPF+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TPF+2]]: [B1.[[#TPF+9]]]
// CHECK-NEXT:  [[#TPF+3]]: [B1.[[#TPF+9]]] = [B1.[[#TPF+1]]]
// CHECK-NEXT:  [[#TPF+4]]: cond
// CHECK-NEXT:  [[#TPF+5]]: [B1.[[#TPF+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TPF+6]]: [B1.[[#TPF+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TPF+7]]: fp
// CHECK-NEXT:  [[#TPF+8]]: rd
// CHECK-NEXT:  [[#TPF+9]]: argc
// CHECK-NEXT:  [[#TPF+10]]: x
// CHECK-NEXT:  [[#TPF+11]]: #pragma omp target parallel for if(parallel: cond) firstprivate(fp) reduction(max: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TPF+3]]];
#pragma omp target parallel for if(parallel:cond) firstprivate(fp) reduction(max:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TPFS:]]:
// CHECK-SAME:  [B1.[[#TPFS+10]]]
// CHECK-NEXT:  [[#TPFS+1]]: [B1.[[#TPFS+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TPFS+2]]: [B1.[[#TPFS+9]]]
// CHECK-NEXT:  [[#TPFS+3]]: [B1.[[#TPFS+9]]] = [B1.[[#TPFS+1]]]
// CHECK-NEXT:  [[#TPFS+4]]: cond
// CHECK-NEXT:  [[#TPFS+5]]: [B1.[[#TPFS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TPFS+6]]: [B1.[[#TPFS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TPFS+7]]: fp
// CHECK-NEXT:  [[#TPFS+8]]: rd
// CHECK-NEXT:  [[#TPFS+9]]: argc
// CHECK-NEXT:  [[#TPFS+10]]: x
// CHECK-NEXT:  [[#TPFS+11]]: #pragma omp target parallel for simd if(target: cond) firstprivate(fp) reduction(*: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TPFS+3]]];
#pragma omp target parallel for simd if(target:cond) firstprivate(fp) reduction(*:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TP:]]:
// CHECK-SAME:  [B1.[[#TP+10]]]
// CHECK-NEXT:  [[#TP+1]]: [B1.[[#TP+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TP+2]]: [B1.[[#TP+9]]]
// CHECK-NEXT:  [[#TP+3]]: [B1.[[#TP+9]]] = [B1.[[#TP+1]]]
// CHECK-NEXT:  [[#TP+4]]: cond
// CHECK-NEXT:  [[#TP+5]]: [B1.[[#TP+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TP+6]]: [B1.[[#TP+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TP+7]]: fp
// CHECK-NEXT:  [[#TP+8]]: rd
// CHECK-NEXT:  [[#TP+9]]: argc
// CHECK-NEXT:  [[#TP+10]]: x
// CHECK-NEXT:  [[#TP+11]]: #pragma omp target parallel if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    [B1.[[#TP+3]]];
#pragma omp target parallel if(cond) firstprivate(fp) reduction(+:rd)
  argc = x;
// CHECK-NEXT:  [[#TSIMD:]]:
// CHECK-SAME:  [B1.[[#TSIMD+10]]]
// CHECK-NEXT:  [[#TSIMD+1]]: [B1.[[#TSIMD+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TSIMD+2]]: [B1.[[#TSIMD+9]]]
// CHECK-NEXT:  [[#TSIMD+3]]: [B1.[[#TSIMD+9]]] = [B1.[[#TSIMD+1]]]
// CHECK-NEXT:  [[#TSIMD+4]]: cond
// CHECK-NEXT:  [[#TSIMD+5]]: [B1.[[#TSIMD+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TSIMD+6]]: [B1.[[#TSIMD+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TSIMD+7]]: fp
// CHECK-NEXT:  [[#TSIMD+8]]: rd
// CHECK-NEXT:  [[#TSIMD+9]]: argc
// CHECK-NEXT:  [[#TSIMD+10]]: x
// CHECK-NEXT:  [[#TSIMD+11]]: #pragma omp target simd if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TSIMD+3]]];
#pragma omp target simd if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TTD:]]:
// CHECK-SAME:  [B1.[[#TTD+10]]]
// CHECK-NEXT:  [[#TTD+1]]: [B1.[[#TTD+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTD+2]]: [B1.[[#TTD+9]]]
// CHECK-NEXT:  [[#TTD+3]]: [B1.[[#TTD+9]]] = [B1.[[#TTD+1]]]
// CHECK-NEXT:  [[#TTD+4]]: cond
// CHECK-NEXT:  [[#TTD+5]]: [B1.[[#TTD+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTD+6]]: [B1.[[#TTD+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TTD+7]]: fp
// CHECK-NEXT:  [[#TTD+8]]: rd
// CHECK-NEXT:  [[#TTD+9]]: argc
// CHECK-NEXT:  [[#TTD+10]]: x
// CHECK-NEXT:  [[#TTD+11]]: #pragma omp target teams distribute if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TTD+3]]];
#pragma omp target teams distribute if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TTDPF:]]:
// CHECK-SAME:  [B1.[[#TTDPF+10]]]
// CHECK-NEXT:  [[#TTDPF+1]]: [B1.[[#TTDPF+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTDPF+2]]: [B1.[[#TTDPF+9]]]
// CHECK-NEXT:  [[#TTDPF+3]]: [B1.[[#TTDPF+9]]] = [B1.[[#TTDPF+1]]]
// CHECK-NEXT:  [[#TTDPF+4]]: cond
// CHECK-NEXT:  [[#TTDPF+5]]: [B1.[[#TTDPF+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTDPF+6]]: [B1.[[#TTDPF+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TTDPF+7]]: fp
// CHECK-NEXT:  [[#TTDPF+8]]: rd
// CHECK-NEXT:  [[#TTDPF+9]]: argc
// CHECK-NEXT:  [[#TTDPF+10]]: x
// CHECK-NEXT:  [[#TTDPF+11]]: #pragma omp target teams distribute parallel for if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TTDPF+3]]];
#pragma omp target teams distribute parallel for if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TTDPFS:]]:
// CHECK-SAME:  [B1.[[#TTDPFS+10]]]
// CHECK-NEXT:  [[#TTDPFS+1]]: [B1.[[#TTDPFS+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTDPFS+2]]: [B1.[[#TTDPFS+9]]]
// CHECK-NEXT:  [[#TTDPFS+3]]: [B1.[[#TTDPFS+9]]] = [B1.[[#TTDPFS+1]]]
// CHECK-NEXT:  [[#TTDPFS+4]]: cond
// CHECK-NEXT:  [[#TTDPFS+5]]: [B1.[[#TTDPFS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTDPFS+6]]: [B1.[[#TTDPFS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TTDPFS+7]]: fp
// CHECK-NEXT:  [[#TTDPFS+8]]: rd
// CHECK-NEXT:  [[#TTDPFS+9]]: argc
// CHECK-NEXT:  [[#TTDPFS+10]]: x
// CHECK-NEXT:  [[#TTDPFS+11]]: #pragma omp target teams distribute parallel for simd if(parallel: cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TTDPFS+3]]];
#pragma omp target teams distribute parallel for simd if(parallel:cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TTDS:]]:
// CHECK-SAME:  [B1.[[#TTDS+10]]]
// CHECK-NEXT:  [[#TTDS+1]]: [B1.[[#TTDS+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTDS+2]]: [B1.[[#TTDS+9]]]
// CHECK-NEXT:  [[#TTDS+3]]: [B1.[[#TTDS+9]]] = [B1.[[#TTDS+1]]]
// CHECK-NEXT:  [[#TTDS+4]]: cond
// CHECK-NEXT:  [[#TTDS+5]]: [B1.[[#TTDS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TTDS+6]]: [B1.[[#TTDS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TTDS+7]]: fp
// CHECK-NEXT:  [[#TTDS+8]]: rd
// CHECK-NEXT:  [[#TTDS+9]]: argc
// CHECK-NEXT:  [[#TTDS+10]]: x
// CHECK-NEXT:  [[#TTDS+11]]: #pragma omp target teams distribute simd if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TTDS+3]]];
#pragma omp target teams distribute simd if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TT:]]:
// CHECK-SAME:  [B1.[[#TT+10]]]
// CHECK-NEXT:  [[#TT+1]]: [B1.[[#TT+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TT+2]]: [B1.[[#TT+9]]]
// CHECK-NEXT:  [[#TT+3]]: [B1.[[#TT+9]]] = [B1.[[#TT+1]]]
// CHECK-NEXT:  [[#TT+4]]: cond
// CHECK-NEXT:  [[#TT+5]]: [B1.[[#TT+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TT+6]]: [B1.[[#TT+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TT+7]]: fp
// CHECK-NEXT:  [[#TT+8]]: rd
// CHECK-NEXT:  [[#TT+9]]: argc
// CHECK-NEXT:  [[#TT+10]]: x
// CHECK-NEXT:  [[#TT+11]]: #pragma omp target teams if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    [B1.[[#TT+3]]];
#pragma omp target teams if(cond) firstprivate(fp) reduction(+:rd)
  argc = x;
// CHECK-NEXT: [[#TU:]]: cond
// CHECK-NEXT: [[#TU+1]]: [B1.[[#TU]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT: [[#TU+2]]: [B1.[[#TU+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT: [[#TU+3]]: #pragma omp target update to(x) if(target update: cond)
#pragma omp target update to(x) if(target update:cond)
// CHECK-NEXT:  [[#TASK:]]:
// CHECK-SAME:  [B1.[[#TASK+9]]]
// CHECK-NEXT:  [[#TASK+1]]: [B1.[[#TASK+9]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TASK+2]]: [B1.[[#TASK+8]]]
// CHECK-NEXT:  [[#TASK+3]]: [B1.[[#TASK+8]]] = [B1.[[#TASK+1]]]
// CHECK-NEXT:  [[#TASK+4]]: cond
// CHECK-NEXT:  [[#TASK+5]]: [B1.[[#TASK+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TASK+6]]: [B1.[[#TASK+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TASK+7]]: fp
// CHECK-NEXT:  [[#TASK+8]]: argc
// CHECK-NEXT:  [[#TASK+9]]: x
// CHECK-NEXT:  [[#TASK+10]]: #pragma omp task if(cond) firstprivate(fp)
// CHECK-NEXT:    [B1.[[#TASK+3]]];
#pragma omp task if(cond) firstprivate(fp)
  argc = x;
// CHECK-NEXT:  [[#TG:]]: x
// CHECK-NEXT:  [[#TG+1]]: [B1.[[#TG]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TG+2]]: argc
// CHECK-NEXT:  [[#TG+3]]: [B1.[[#TG+2]]] = [B1.[[#TG+1]]]
// CHECK-NEXT:  [[#TG+4]]: #pragma omp taskgroup
// CHECK-NEXT:    [B1.[[#TG+3]]];
#pragma omp taskgroup
  argc = x;
// CHECK-NEXT:  [[#TL:]]:
// CHECK-SAME:  [B1.[[#TL+10]]]
// CHECK-NEXT:  [[#TL+1]]: [B1.[[#TL+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TL+2]]: [B1.[[#TL+9]]]
// CHECK-NEXT:  [[#TL+3]]: [B1.[[#TL+9]]] = [B1.[[#TL+1]]]
// CHECK-NEXT:  [[#TL+4]]: cond
// CHECK-NEXT:  [[#TL+5]]: [B1.[[#TL+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TL+6]]: [B1.[[#TL+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TL+7]]: fp
// CHECK-NEXT:  [[#TL+8]]: rd
// CHECK-NEXT:  [[#TL+9]]: argc
// CHECK-NEXT:  [[#TL+10]]: x
// CHECK-NEXT:  [[#TL+11]]: #pragma omp taskloop if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TL+3]]];
#pragma omp taskloop if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TLS:]]:
// CHECK-SAME:  [B1.[[#TLS+10]]]
// CHECK-NEXT:  [[#TLS+1]]: [B1.[[#TLS+10]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TLS+2]]: [B1.[[#TLS+9]]]
// CHECK-NEXT:  [[#TLS+3]]: [B1.[[#TLS+9]]] = [B1.[[#TLS+1]]]
// CHECK-NEXT:  [[#TLS+4]]: cond
// CHECK-NEXT:  [[#TLS+5]]: [B1.[[#TLS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TLS+6]]: [B1.[[#TLS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TLS+7]]: fp
// CHECK-NEXT:  [[#TLS+8]]: rd
// CHECK-NEXT:  [[#TLS+9]]: argc
// CHECK-NEXT:  [[#TLS+10]]: x
// CHECK-NEXT:  [[#TLS+11]]: #pragma omp taskloop simd if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TLS+3]]];
#pragma omp taskloop simd if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [[#TDPF:]]: x
// CHECK-NEXT:  [[#TDPF+1]]: [B1.[[#TDPF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TDPF+2]]: argc
// CHECK-NEXT:  [[#TDPF+3]]: [B1.[[#TDPF+2]]] = [B1.[[#TDPF+1]]]
// CHECK-NEXT:  [[#TDPF+4]]: cond
// CHECK-NEXT:  [[#TDPF+5]]: [B1.[[#TDPF+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TDPF+6]]: [B1.[[#TDPF+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TDPF+7]]: [B1.[[#TDPF+10]]]
// CHECK-NEXT:  [[#TDPF+8]]: [B1.[[#TDPF+11]]]
// CHECK-NEXT:  [[#TDPF+9]]: #pragma omp teams distribute parallel for if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TDPF+3]]];
// CHECK-NEXT:  [[#TDPF+10]]: fp
// CHECK-NEXT:  [[#TDPF+11]]: rd
// CHECK-NEXT:  [[#TDPF+12]]: argc
// CHECK-NEXT:  [[#TDPF+13]]: x
// CHECK-NEXT:  [[#TDPF+14]]: cond
// CHECK-NEXT:  [[#TDPF+15]]: #pragma omp target
#pragma omp target
#pragma omp teams distribute parallel for if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [B1.[[#TDPF+9]]] [[#TDPFS:]]: x
// CHECK-NEXT:  [[#TDPFS+1]]: [B1.[[#TDPFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TDPFS+2]]: argc
// CHECK-NEXT:  [[#TDPFS+3]]: [B1.[[#TDPFS+2]]] = [B1.[[#TDPFS+1]]]
// CHECK-NEXT:  [[#TDPFS+4]]: cond
// CHECK-NEXT:  [[#TDPFS+5]]: [B1.[[#TDPFS+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TDPFS+6]]: [B1.[[#TDPFS+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TDPFS+7]]: [B1.[[#TDPFS+10]]]
// CHECK-NEXT:  [[#TDPFS+8]]: [B1.[[#TDPFS+11]]]
// CHECK-NEXT:  [[#TDPFS+9]]: #pragma omp teams distribute parallel for simd if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TDPFS+3]]];
// CHECK-NEXT:  [[#TDPFS+10]]: fp
// CHECK-NEXT:  [[#TDPFS+11]]: rd
// CHECK-NEXT:  [[#TDPFS+12]]: argc
// CHECK-NEXT:  [[#TDPFS+13]]: x
// CHECK-NEXT:  [[#TDPFS+14]]: cond
// CHECK-NEXT:  [[#TDPFS+15]]: #pragma omp target
#pragma omp target
#pragma omp teams distribute parallel for simd if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [B1.[[#TDPFS+9]]] [[#TDS:]]: x
// CHECK-NEXT:  [[#TDS+1]]: [B1.[[#TDS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TDS+2]]: argc
// CHECK-NEXT:  [[#TDS+3]]: [B1.[[#TDS+2]]] = [B1.[[#TDS+1]]]
// CHECK-NEXT:  [[#TDS+4]]: [B1.[[#TDS+7]]]
// CHECK-NEXT:  [[#TDS+5]]: [B1.[[#TDS+8]]]
// CHECK-NEXT:  [[#TDS+6]]: #pragma omp teams distribute simd firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    for (int i = 0; i < 10; ++i)
// CHECK-NEXT:        [B1.[[#TDS+3]]];
// CHECK-NEXT:  [[#TDS+7]]: fp
// CHECK-NEXT:  [[#TDS+8]]: rd
// CHECK-NEXT:  [[#TDS+9]]: argc
// CHECK-NEXT:  [[#TDS+10]]: x
// CHECK-NEXT:  [[#TDS+11]]: #pragma omp target
#pragma omp target
#pragma omp teams distribute simd firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
// CHECK-NEXT:  [B1.[[#TDS+6]]] [[#TEAMS:]]: x
// CHECK-NEXT:  [[#TEAMS+1]]: [B1.[[#TEAMS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TEAMS+2]]: argc
// CHECK-NEXT:  [[#TEAMS+3]]: [B1.[[#TEAMS+2]]] = [B1.[[#TEAMS+1]]]
// CHECK-NEXT:  [[#TEAMS+4]]: [B1.[[#TEAMS+7]]]
// CHECK-NEXT:  [[#TEAMS+5]]: [B1.[[#TEAMS+8]]]
// CHECK-NEXT:  [[#TEAMS+6]]: #pragma omp teams firstprivate(fp) reduction(+: rd)
// CHECK-NEXT:    [B1.[[#TEAMS+3]]];
// CHECK-NEXT:  [[#TEAMS+7]]: fp
// CHECK-NEXT:  [[#TEAMS+8]]: rd
// CHECK-NEXT:  [[#TEAMS+9]]: argc
// CHECK-NEXT:  [[#TEAMS+10]]: x
// CHECK-NEXT:  [[#TEAMS+11]]: #pragma omp target
#pragma omp target
#pragma omp teams firstprivate(fp) reduction(+:rd)
  argc = x;
// CHECK-NEXT:  [B1.[[#TEAMS+6]]]   Preds
}

