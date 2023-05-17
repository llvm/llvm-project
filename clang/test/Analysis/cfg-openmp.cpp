// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG %s 2>&1 -fopenmp -fopenmp-version=45 | FileCheck %s

// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCFG %s 2>&1 -fopenmp -fopenmp-version=51 | FileCheck %s --check-prefix=OMP51

#if _OPENMP == 202011

// OMP51-LABEL:  void target_has_device_addr(int argc)
void target_has_device_addr(int argc) {
// OMP51:   [B1]
// OMP51-NEXT:   [[#TTD:]]: 5
// OMP51-NEXT:   [[#TTD+1]]: int x = 5;
// OMP51-NEXT:   [[#TTD+2]]: x
// OMP51-NEXT:   [[#TTD+3]]: [B1.[[#TTD+2]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-NEXT:   [[#TTD+4]]: [B1.[[#TTD+6]]]
// OMP51-NEXT:   [[#TTD+5]]: [B1.[[#TTD+6]]] = [B1.[[#TTD+3]]]
// OMP51-NEXT:   [[#TTD+6]]: argc
// OMP51-NEXT:   [[#TTD+7]]: #pragma omp target has_device_addr(x)
// OMP51-NEXT:   [B1.[[#TTD+5]]]
  int x = 5;
#pragma omp target has_device_addr(x)
   argc = x;
}
// OMP51-LABEL: void target_s_has_device_addr(int argc)
void target_s_has_device_addr(int argc) {
  int x, cond, fp, rd, lin, step, map;
// OMP51-DAG:  [B3]
// OMP51-DAG: [[#TSB:]]: x
// OMP51-DAG: [[#TSB+1]]: [B3.[[#TSB]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TSB+2]]: argc
// OMP51-DAG: [[#TSB+3]]: [B3.[[#TSB+2]]] = [B3.[[#TSB+1]]]
// OMP51-DAG:  [B1]
// OMP51-DAG: [[#TS:]]: cond
// OMP51-DAG: [[#TS+1]]: [B1.[[#TS]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TS+2]]: [B1.[[#TS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// OMP51-DAG: [[#TS+3]]: fp
// OMP51-DAG: [[#TS+4]]: rd
// OMP51-DAG: [[#TS+5]]: lin
// OMP51-DAG: [[#TS+6]]: step
// OMP51-DAG: [[#TS+7]]: [B1.[[#TS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TS+8]]: [B3.[[#TSB+2]]]
// OMP51-DAG: [[#TS+9]]: [B3.[[#TSB]]]
// OMP51-DAG: [[#TS+10]]: #pragma omp target simd if(cond) firstprivate(fp) reduction(+: rd) linear(lin: step) has_device_addr(map)
// OMP51-DAG:    for (int i = 0;
// OMP51-DAG: [B3.[[#TSB+3]]];
#pragma omp target simd if(cond) firstprivate(fp) reduction(+:rd) linear(lin: step) has_device_addr(map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}
// OMP51-LABEL: void target_t_l_has_device_addr(int argc)
void target_t_l_has_device_addr(int argc) {
int x, cond, fp, rd, map;
// OMP51-DAG: [B3]
// OMP51-DAG: [[#TTDB:]]: x
// OMP51-DAG: [[#TTDB+1]]: [B3.[[#TTDB]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TTDB+2]]: argc
// OMP51-DAG: [[#TTDB+3]]: [B3.[[#TTDB+2]]] = [B3.[[#TTDB+1]]]
// OMP51-DAG: [B1]
// OMP51-DAG: [[#TTD:]]: cond
// OMP51-DAG: [[#TTD+1]]: [B1.[[#TTD]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TTD+2]]: [B1.[[#TTD+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// OMP51-DAG: [[#TTD+3]]: fp
// OMP51-DAG: [[#TTD+4]]: rd
// OMP51-DAG: [[#TTD+5]]: [B3.[[#TTDB+2]]]
// OMP51-DAG: [[#TTD+6]]: [B3.[[#TTDB]]]
// OMP51-DAG: [[#TTD+7]]:  #pragma omp target teams loop if(cond) firstprivate(fp) reduction(+: rd) has_device_addr(map)
// OMP51-DAG: for (int i = 0;
// OMP51-DAG:   [B3.[[#TTDB+3]]];
#pragma omp target teams loop if(cond) firstprivate(fp) reduction(+:rd) has_device_addr(map)
   for (int i = 0; i <10; ++i)
     argc = x;
}
// OMP51-LABEL:  void target_p_l_has_device_addr(int argc)
void target_p_l_has_device_addr(int argc) {
int x, cond, fp, rd, map;
#pragma omp target parallel loop if(cond) firstprivate(fp) reduction(+:rd) has_device_addr(map)
// OMP51-DAG: [B3]
// OMP51-DAG: [[#TTDB:]]: x
// OMP51-DAG: [[#TTDB+1]]: [B3.[[#TTDB]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TTDB+2]]: argc
// OMP51-DAG: [[#TTDB+3]]: [B3.[[#TTDB+2]]] = [B3.[[#TTDB+1]]]
// OMP51-DAG: [B1]
// OMP51-DAG: [[#TTD:]]: cond
// OMP51-DAG: [[#TTD+1]]: [B1.[[#TTD]]] (ImplicitCastExpr, LValueToRValue, int)
// OMP51-DAG: [[#TTD+2]]: [B1.[[#TTD+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// OMP51-DAG: [[#TTD+3]]: fp
// OMP51-DAG: [[#TTD+4]]: rd
// OMP51-DAG: [[#TTD+5]]: [B3.[[#TTDB+2]]]
// OMP51-DAG: [[#TTD+6]]: [B3.[[#TTDB]]]
// OMP51-DAG: [[#TTD+7]]: #pragma omp target parallel loop if(cond) firstprivate(fp) reduction(+: rd) has_device_addr(map)
// OMP51-DAG: for (int i = 0;
// OMP51-DAG:   [B3.[[#TTDB+3]]];
  for (int i = 0; i < 10; ++i)
    argc = x;
}
struct SomeKernel {
  int targetDev;
  float devPtr;
  SomeKernel();
  ~SomeKernel();
// OMP51-LABEL: template<> void apply<32U>()
  template<unsigned int nRHS>
  void apply() {
// OMP51-DAG: [B1]
// OMP51-DAG: [[#TTD:]]: 10
// OMP51-DAG: [[#TTD+1]]: [B1.[[#TTD:]]] (ImplicitCastExpr, IntegralToFloating, float)
// OMP51-DAG: [[#TTD+2]]: this
// OMP51-DAG: [[#TTD+3]]: [B1.[[#TTD+2]]]->devPtr
// OMP51-DAG: [[#TTD+4]]: [B1.[[#TTD+3]]] = [B1.[[#TTD+1]]]
// OMP51-DAG: [[#TTD+5]]: #pragma omp target has_device_addr(this->devPtr) device(this->targetDev)
// OMP51-DAG:    {
// OMP51-DAG:    [B1.[[#TTD+4]]];
    #pragma omp target has_device_addr(devPtr) device(targetDev)
    {
      devPtr = 10;
    }
  }
};
void use_template() {
  SomeKernel aKern;
  aKern.apply<32>();
}
#else // _OPENMP

// CHECK-LABEL:  void xxx(int argc)
void xxx(int argc) {
// CHECK:        [B1]
// CHECK-NEXT:   1: int x;
// CHECK-NEXT:   2: int cond;
// CHECK-NEXT:   3: int fp;
// CHECK-NEXT:   4: int rd;
// CHECK-NEXT:   5: int lin;
// CHECK-NEXT:   6: int step;
// CHECK-NEXT:   7: int map;
  int x, cond, fp, rd, lin, step, map;
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
// CHECK-NEXT:  [[#MASTER:]]: x
// CHECK-NEXT:  [[#MASTER+1]]: [B1.[[#MASTER]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#MASTER+2]]: argc
// CHECK-NEXT:  [[#MASTER+3]]: [B1.[[#MASTER+2]]] = [B1.[[#MASTER+1]]]
// CHECK-NEXT:  [[#MASTER+4]]: #pragma omp master
// CHECK-NEXT:    [B1.[[#MASTER+3]]];
#pragma omp master
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
// CHECK-NEXT:  [[#TP:]]:
// CHECK-SAME:  [B1.[[#TP+11]]]
// CHECK-NEXT:  [[#TP+1]]: [B1.[[#TP+11]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TP+2]]: [B1.[[#TP+10]]]
// CHECK-NEXT:  [[#TP+3]]: [B1.[[#TP+10]]] = [B1.[[#TP+1]]]
// CHECK-NEXT:  [[#TP+4]]: cond
// CHECK-NEXT:  [[#TP+5]]: [B1.[[#TP+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TP+6]]: [B1.[[#TP+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TP+7]]: fp
// CHECK-NEXT:  [[#TP+8]]: rd
// CHECK-NEXT:  [[#TP+9]]: map
// CHECK-NEXT:  [[#TP+10]]: argc
// CHECK-NEXT:  [[#TP+11]]: x
// CHECK-NEXT:  [[#TP+12]]: #pragma omp target parallel if(cond) firstprivate(fp) reduction(+: rd) map(to: map)
// CHECK-NEXT:    [B1.[[#TP+3]]];
#pragma omp target parallel if(cond) firstprivate(fp) reduction(+:rd) map(to:map)
  argc = x;
// CHECK-NEXT:  [[#TT:]]:
// CHECK-SAME:  [B1.[[#TT+11]]]
// CHECK-NEXT:  [[#TT+1]]: [B1.[[#TT+11]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TT+2]]: [B1.[[#TT+10]]]
// CHECK-NEXT:  [[#TT+3]]: [B1.[[#TT+10]]] = [B1.[[#TT+1]]]
// CHECK-NEXT:  [[#TT+4]]: cond
// CHECK-NEXT:  [[#TT+5]]: [B1.[[#TT+4]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:  [[#TT+6]]: [B1.[[#TT+5]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-NEXT:  [[#TT+7]]: fp
// CHECK-NEXT:  [[#TT+8]]: rd
// CHECK-NEXT:  [[#TT+9]]: map
// CHECK-NEXT:  [[#TT+10]]: argc
// CHECK-NEXT:  [[#TT+11]]: x
// CHECK-NEXT:  [[#TT+12]]: #pragma omp target teams if(cond) firstprivate(fp) reduction(+: rd) map(tofrom: map)
// CHECK-NEXT:    [B1.[[#TT+3]]];
#pragma omp target teams if(cond) firstprivate(fp) reduction(+:rd) map(tofrom:map)
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
// CHECK-NEXT:  [[#TEAMS:]]: x
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

// CHECK-LABEL:  void dpf(int argc)
void dpf(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#DPFB:]]: x
// CHECK-DAG:  [[#DPFB+1]]: [B3.[[#DPFB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#DPFB+2]]: argc
// CHECK-DAG:  [[#DPFB+3]]: [B3.[[#DPFB+2]]] = [B3.[[#DPFB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#DPF:]]: cond
// CHECK-DAG:  [[#DPF+1]]: [B1.[[#DPF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#DPF+2]]: [B1.[[#DPF+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#DPF+3]]: fp
// CHECK-DAG:  [[#DPF+4]]: rd
// CHECK-DAG:  [[#DPF+5]]: #pragma omp distribute parallel for if(parallel: cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#DPFB+3]]];
#pragma omp distribute parallel for if(parallel:cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void dpfs(int argc)
void dpfs(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#DPFSB:]]: x
// CHECK-DAG:  [[#DPFSB+1]]: [B3.[[#DPFSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#DPFSB+2]]: argc
// CHECK-DAG:  [[#DPFSB+3]]: [B3.[[#DPFSB+2]]] = [B3.[[#DPFSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#DPFS:]]: cond
// CHECK-DAG:  [[#DPFS+1]]: [B1.[[#DPFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#DPFS+2]]: [B1.[[#DPFS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#DPFS+3]]: fp
// CHECK-DAG:  [[#DPFS+4]]: rd
// CHECK-DAG:  [[#DPFS+5]]: #pragma omp distribute parallel for simd if(cond) firstprivate(fp) reduction(-: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#DPFSB+3]]];
#pragma omp distribute parallel for simd if(cond)  firstprivate(fp) reduction(-:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ds(int argc)
void ds(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#DSB:]]: x
// CHECK-DAG:  [[#DSB+1]]: [B3.[[#DSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#DSB+2]]: argc
// CHECK-DAG:  [[#DSB+3]]: [B3.[[#DSB+2]]] = [B3.[[#DSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#DS:]]: #pragma omp distribute simd
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#DSB+3]]];
#pragma omp distribute simd
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void for_fn(int argc)
void for_fn(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#FORB:]]: x
// CHECK-DAG:  [[#FORB+1]]: [B3.[[#FORB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#FORB+2]]: argc
// CHECK-DAG:  [[#FORB+3]]: [B3.[[#FORB+2]]] = [B3.[[#FORB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#FOR:]]: lin
// CHECK-DAG:  [[#FOR+1]]: step
// CHECK-DAG:  [[#FOR+2]]: [B1.[[#FOR+1]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#FOR+3]]: #pragma omp for linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#FORB+3]]];
#pragma omp for linear(lin : step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void fs(int argc)
void fs(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#FSB:]]: x
// CHECK-DAG:  [[#FSB+1]]: [B3.[[#FSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#FSB+2]]: argc
// CHECK-DAG:  [[#FSB+3]]: [B3.[[#FSB+2]]] = [B3.[[#FSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#FS:]]: lin
// CHECK-DAG:  [[#FS+1]]: step
// CHECK-DAG:  [[#FS+2]]: [B1.[[#FS+1]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#FS+3]]: #pragma omp for simd linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#FSB+3]]];
#pragma omp for simd linear(lin: step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ord(int argc)
void ord(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#ORDB:]]: x
// CHECK-DAG:  [[#ORDB+1]]: [B3.[[#ORDB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#ORDB+2]]: argc
// CHECK-DAG:  [[#ORDB+3]]: [B3.[[#ORDB+2]]] = [B3.[[#ORDB+1]]]
// CHECK-DAG:  [[#ORDB+4]]: #pragma omp ordered
// CHECK-DAG:    [B3.[[#ORDB+3]]];
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#ORD:]]: #pragma omp for ordered
// CHECK-DAG:    for (int i = 0
// CHECK-DAG:[B3.[[#ORDB+4]]]    }
#pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered
    argc = x;
  }
}

// CHECK-LABEL:  void pf(int argc)
void pf(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#PFB:]]: x
// CHECK-DAG:  [[#PFB+1]]: [B3.[[#PFB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PFB+2]]: argc
// CHECK-DAG:  [[#PFB+3]]: [B3.[[#PFB+2]]] = [B3.[[#PFB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#PF:]]: cond
// CHECK-DAG:  [[#PF+1]]: [B1.[[#PF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PF+2]]: [B1.[[#PF+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#PF+3]]: fp
// CHECK-DAG:  [[#PF+4]]: rd
// CHECK-DAG:  [[#PF+5]]: lin
// CHECK-DAG:  [[#PF+6]]: step
// CHECK-DAG:  [[#PF+7]]: [B1.[[#PF+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PF+8]]: #pragma omp parallel for if(cond) firstprivate(fp) reduction(&: rd) linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#PFB+3]]];
#pragma omp parallel for if(cond) firstprivate(fp) reduction(&:rd) linear(lin: step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void pfs(int argc)
void pfs(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#PFSB:]]: x
// CHECK-DAG:  [[#PFSB+1]]: [B3.[[#PFSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PFSB+2]]: argc
// CHECK-DAG:  [[#PFSB+3]]: [B3.[[#PFSB+2]]] = [B3.[[#PFSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#PFS:]]: cond
// CHECK-DAG:  [[#PFS+1]]: [B1.[[#PFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PFS+2]]: [B1.[[#PFS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#PFS+3]]: fp
// CHECK-DAG:  [[#PFS+4]]: rd
// CHECK-DAG:  [[#PFS+5]]: lin
// CHECK-DAG:  [[#PFS+6]]: step
// CHECK-DAG:  [[#PFS+7]]: [B1.[[#PFS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PFS+8]]: #pragma omp parallel for simd if(cond) firstprivate(fp) reduction(|: rd) linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#PFSB+3]]];
#pragma omp parallel for simd if(cond) firstprivate(fp) reduction(|:rd) linear(lin: step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void simd(int argc)
void simd(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#SIMDB:]]: x
// CHECK-DAG:  [[#SIMDB+1]]: [B3.[[#SIMDB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#SIMDB+2]]: argc
// CHECK-DAG:  [[#SIMDB+3]]: [B3.[[#SIMDB+2]]] = [B3.[[#SIMDB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#SIMD:]]: lin
// CHECK-DAG:  [[#SIMD+1]]: step
// CHECK-DAG:  [[#SIMD+2]]: [B1.[[#SIMD+1]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#SIMD+3]]: #pragma omp simd linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#SIMDB+3]]];
#pragma omp simd linear(lin: step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tpf(int argc)
void tpf(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TPFB:]]: x
// CHECK-DAG:  [[#TPFB+1]]: [B3.[[#TPFB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TPFB+2]]: argc
// CHECK-DAG:  [[#TPFB+3]]: [B3.[[#TPFB+2]]] = [B3.[[#TPFB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TPF:]]: cond
// CHECK-DAG:  [[#TPF+1]]: [B1.[[#TPF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TPF+2]]: [B1.[[#TPF+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TPF+3]]: fp
// CHECK-DAG:  [[#TPF+4]]: rd
// CHECK-DAG:  [[#TPF+5]]: lin
// CHECK-DAG:  [[#TPF+6]]: step
// CHECK-DAG:  [[#TPF+7]]: [B1.[[#TPF+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TPF+8]]: map
// CHECK-DAG:  [[#TPF+9]]: [B3.[[#TPFB+2]]]
// CHECK-DAG:  [[#TPF+10]]: [B3.[[#TPFB]]]
// CHECK-DAG:  [[#TPF+11]]: #pragma omp target parallel for if(parallel: cond) firstprivate(fp) reduction(max: rd) linear(lin: step) map(tofrom: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TPFB+3]]];
#pragma omp target parallel for if(parallel:cond) firstprivate(fp) reduction(max:rd) linear(lin: step) map(map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tpfs(int argc)
void tpfs(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TPFSB:]]: x
// CHECK-DAG:  [[#TPFSB+1]]: [B3.[[#TPFSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TPFSB+2]]: argc
// CHECK-DAG:  [[#TPFSB+3]]: [B3.[[#TPFSB+2]]] = [B3.[[#TPFSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TPFS:]]: cond
// CHECK-DAG:  [[#TPFS+1]]: [B1.[[#TPFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TPFS+2]]: [B1.[[#TPFS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TPFS+3]]: fp
// CHECK-DAG:  [[#TPFS+4]]: rd
// CHECK-DAG:  [[#TPFS+5]]: lin
// CHECK-DAG:  [[#TPFS+6]]: step
// CHECK-DAG:  [[#TPFS+7]]: [B1.[[#TPFS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TPFS+8]]: map
// CHECK-DAG:  [[#TPFS+9]]: [B3.[[#TPFSB+2]]]
// CHECK-DAG:  [[#TPFS+10]]: [B3.[[#TPFSB]]]
// CHECK-DAG:  [[#TPFS+11]]: #pragma omp target parallel for simd if(target: cond) firstprivate(fp) reduction(*: rd) linear(lin: step) map(tofrom: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TPFSB+3]]];
#pragma omp target parallel for simd if(target:cond) firstprivate(fp) reduction(*:rd) linear(lin: step) map(tofrom:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ts(int argc)
void ts(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TSB:]]: x
// CHECK-DAG:  [[#TSB+1]]: [B3.[[#TSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TSB+2]]: argc
// CHECK-DAG:  [[#TSB+3]]: [B3.[[#TSB+2]]] = [B3.[[#TSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TS:]]: cond
// CHECK-DAG:  [[#TS+1]]: [B1.[[#TS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TS+2]]: [B1.[[#TS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TS+3]]: fp
// CHECK-DAG:  [[#TS+4]]: rd
// CHECK-DAG:  [[#TS+5]]: lin
// CHECK-DAG:  [[#TS+6]]: step
// CHECK-DAG:  [[#TS+7]]: [B1.[[#TS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TS+8]]: [B3.[[#TSB+2]]]
// CHECK-DAG:  [[#TS+9]]: [B3.[[#TSB]]]
// CHECK-DAG:  [[#TS+10]]: #pragma omp target simd if(cond) firstprivate(fp) reduction(+: rd) linear(lin: step) map(alloc: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TSB+3]]];
#pragma omp target simd if(cond) firstprivate(fp) reduction(+:rd) linear(lin: step) map(alloc:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ttd(int argc)
void ttd(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TTDB:]]: x
// CHECK-DAG:  [[#TTDB+1]]: [B3.[[#TTDB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDB+2]]: argc
// CHECK-DAG:  [[#TTDB+3]]: [B3.[[#TTDB+2]]] = [B3.[[#TTDB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TTD:]]: cond
// CHECK-DAG:  [[#TTD+1]]: [B1.[[#TTD]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTD+2]]: [B1.[[#TTD+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TTD+3]]: fp
// CHECK-DAG:  [[#TTD+4]]: rd
// CHECK-DAG:  [[#TTD+5]]: [B3.[[#TTDB+2]]]
// CHECK-DAG:  [[#TTD+6]]: [B3.[[#TTDB]]]
// CHECK-DAG:  [[#TTD+7]]: #pragma omp target teams distribute if(cond) firstprivate(fp) reduction(+: rd) map(alloc: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TTDB+3]]];
#pragma omp target teams distribute if(cond) firstprivate(fp) reduction(+:rd) map(alloc:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ttdpf(int argc)
void ttdpf(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TTDPFB:]]: x
// CHECK-DAG:  [[#TTDPFB+1]]: [B3.[[#TTDPFB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDPFB+2]]: argc
// CHECK-DAG:  [[#TTDPFB+3]]: [B3.[[#TTDPFB+2]]] = [B3.[[#TTDPFB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TTDPF:]]: cond
// CHECK-DAG:  [[#TTDPF+1]]: [B1.[[#TTDPF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDPF+2]]: [B1.[[#TTDPF+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TTDPF+3]]: fp
// CHECK-DAG:  [[#TTDPF+4]]: rd
// CHECK-DAG:  [[#TTDPF+5]]: [B3.[[#TTDPFB+2]]]
// CHECK-DAG:  [[#TTDPF+6]]: [B3.[[#TTDPFB]]]
// CHECK-DAG:  [[#TTDPF+7]]: #pragma omp target teams distribute parallel for if(cond) firstprivate(fp) reduction(+: rd) map(alloc: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TTDPFB+3]]];
#pragma omp target teams distribute parallel for if(cond) firstprivate(fp) reduction(+:rd) map(alloc:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ttdpfs(int argc)
void ttdpfs(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TTDPFSB:]]: x
// CHECK-DAG:  [[#TTDPFSB+1]]: [B3.[[#TTDPFSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDPFSB+2]]: argc
// CHECK-DAG:  [[#TTDPFSB+3]]: [B3.[[#TTDPFSB+2]]] = [B3.[[#TTDPFSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TTDPFS:]]: cond
// CHECK-DAG:  [[#TTDPFS+1]]: [B1.[[#TTDPFS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDPFS+2]]: [B1.[[#TTDPFS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TTDPFS+3]]: fp
// CHECK-DAG:  [[#TTDPFS+4]]: rd
// CHECK-DAG:  [[#TTDPFS+5]]: [B3.[[#TTDPFSB+2]]]
// CHECK-DAG:  [[#TTDPFS+6]]: [B3.[[#TTDPFSB]]]
// CHECK-DAG:  [[#TTDPFS+7]]: #pragma omp target teams distribute parallel for simd if(parallel: cond) firstprivate(fp) reduction(+: rd) map(from: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TTDPFSB+3]]];
#pragma omp target teams distribute parallel for simd if(parallel:cond) firstprivate(fp) reduction(+:rd) map(from:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void ttds(int argc)
void ttds(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TTDSB:]]: x
// CHECK-DAG:  [[#TTDSB+1]]: [B3.[[#TTDSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDSB+2]]: argc
// CHECK-DAG:  [[#TTDSB+3]]: [B3.[[#TTDSB+2]]] = [B3.[[#TTDSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TTDS:]]: cond
// CHECK-DAG:  [[#TTDS+1]]: [B1.[[#TTDS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDS+2]]: [B1.[[#TTDS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TTDS+3]]: fp
// CHECK-DAG:  [[#TTDS+4]]: rd
// CHECK-DAG:  [[#TTDS+5]]: map
// CHECK-DAG:  [[#TTDS+6]]: [B3.[[#TTDSB+2]]]
// CHECK-DAG:  [[#TTDS+7]]: [B3.[[#TTDSB]]]
// CHECK-DAG:  [[#TTDS+8]]: #pragma omp target teams distribute simd if(cond) firstprivate(fp) reduction(+: rd) map(tofrom: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TTDPFSB+3]]];
#pragma omp target teams distribute simd if(cond) firstprivate(fp) reduction(+:rd) map(map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tl(int argc)
void tl(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TLB:]]: x
// CHECK-DAG:  [[#TLB+1]]: [B3.[[#TLB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TLB+2]]: argc
// CHECK-DAG:  [[#TLB+3]]: [B3.[[#TLB+2]]] = [B3.[[#TLB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TL:]]: cond
// CHECK-DAG:  [[#TL+1]]: [B1.[[#TL]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TL+2]]: [B1.[[#TL+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TL+3]]: fp
// CHECK-DAG:  [[#TL+4]]: rd
// CHECK-DAG:  [[#TL+5]]: [B3.[[#TLB+2]]]
// CHECK-DAG:  [[#TL+6]]: [B3.[[#TLB]]]
// CHECK-DAG:  [[#TL+7]]: #pragma omp taskloop if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TLB+3]]];
#pragma omp taskloop if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void maskedtaskloop(int argc)
void maskedtaskloop(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#MTLB:]]: x
// CHECK-DAG:  [[#MTLB+1]]: [B3.[[#MTLB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#MTLB+2]]: argc
// CHECK-DAG:  [[#MTLB+3]]: [B3.[[#MTLB+2]]] = [B3.[[#MTLB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#MTL:]]: cond
// CHECK-DAG:  [[#MTL+1]]: [B1.[[#MTL]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#MTL+2]]: [B1.[[#MTL+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#MTL+3]]: fp
// CHECK-DAG:  [[#MTL+4]]: rd
// CHECK-DAG:  [[#MTL+5]]: [B3.[[#MTLB+2]]]
// CHECK-DAG:  [[#MTL+6]]: [B3.[[#MTLB]]]
// CHECK-DAG:  [[#MTL+7]]: #pragma omp masked taskloop if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#MTLB+3]]];
#pragma omp masked taskloop if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void parallelmaskedtaskloop(int argc)
void parallelmaskedtaskloop(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#PMTLB:]]: x
// CHECK-DAG:  [[#PMTLB+1]]: [B3.[[#PMTLB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PMTLB+2]]: argc
// CHECK-DAG:  [[#PMTLB+3]]: [B3.[[#PMTLB+2]]] = [B3.[[#PMTLB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#PMTL:]]: cond
// CHECK-DAG:  [[#PMTL+1]]: [B1.[[#PMTL]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PMTL+2]]: [B1.[[#PMTL+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#PMTL+3]]: fp
// CHECK-DAG:  [[#PMTL+4]]: rd
// CHECK-DAG:  [[#PMTL+5]]: [B3.[[#PMTLB+2]]]
// CHECK-DAG:  [[#PMTL+6]]: [B3.[[#PMTLB]]]
// CHECK-DAG:  [[#PMTL+7]]: #pragma omp parallel masked taskloop if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#PMTLB+3]]];
#pragma omp parallel masked taskloop if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void parallelmaskedtasksimdloop(int argc)
void parallelmaskedtasksimdloop(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#PMTLSB:]]: x
// CHECK-DAG:  [[#PMTLSB+1]]: [B3.[[#PMTLSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PMTLSB+2]]: argc
// CHECK-DAG:  [[#PMTLSB+3]]: [B3.[[#PMTLSB+2]]] = [B3.[[#PMTLSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#PMTLS:]]: cond
// CHECK-DAG:  [[#PMTLS+1]]: [B1.[[#PMTLS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PMTLS+2]]: [B1.[[#PMTLS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#PMTLS+3]]: fp
// CHECK-DAG:  [[#PMTLS+4]]: rd
// CHECK-DAG:  [[#PMTLS+5]]: [B3.[[#PMTLSB+2]]]
// CHECK-DAG:  [[#PMTLS+6]]: [B3.[[#PMTLSB]]]
// CHECK-DAG:  [[#PMTLS+7]]: #pragma omp parallel masked taskloop simd if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#PMTLSB+3]]];
#pragma omp parallel masked taskloop simd if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tls(int argc)
void tls(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TLSB:]]: x
// CHECK-DAG:  [[#TLSB+1]]: [B3.[[#TLSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TLSB+2]]: argc
// CHECK-DAG:  [[#TLSB+3]]: [B3.[[#TLSB+2]]] = [B3.[[#TLSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TLS:]]: cond
// CHECK-DAG:  [[#TLS+1]]: [B1.[[#TLS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TLS+2]]: [B1.[[#TLS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TLS+3]]: fp
// CHECK-DAG:  [[#TLS+4]]: rd
// CHECK-DAG:  [[#TLS+5]]: lin
// CHECK-DAG:  [[#TLS+6]]: step
// CHECK-DAG:  [[#TLS+7]]: [B1.[[#TLS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TLS+8]]: [B3.[[#TLSB+2]]]
// CHECK-DAG:  [[#TLS+9]]: [B3.[[#TLSB]]]
// CHECK-DAG:  [[#TLS+10]]: #pragma omp taskloop simd if(cond) firstprivate(fp) reduction(+: rd) linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TLSB+3]]];
#pragma omp taskloop simd if(cond) firstprivate(fp) reduction(+:rd) linear(lin: step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void maskedtaskloopsimd(int argc)
void maskedtaskloopsimd(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#MTLSB:]]: x
// CHECK-DAG:  [[#MTLSB+1]]: [B3.[[#MTLSB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#MTLSB+2]]: argc
// CHECK-DAG:  [[#MTLSB+3]]: [B3.[[#MTLSB+2]]] = [B3.[[#MTLSB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#MTLS:]]: cond
// CHECK-DAG:  [[#MTLS+1]]: [B1.[[#MTLS]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#MTLS+2]]: [B1.[[#MTLS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#MTLS+3]]: fp
// CHECK-DAG:  [[#MTLS+4]]: rd
// CHECK-DAG:  [[#MTLS+5]]: lin
// CHECK-DAG:  [[#MTLS+6]]: step
// CHECK-DAG:  [[#MTLS+7]]: [B1.[[#MTLS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#MTLS+8]]: [B3.[[#MTLSB+2]]]
// CHECK-DAG:  [[#MTLS+9]]: [B3.[[#MTLSB]]]
// CHECK-DAG:  [[#MTLS+10]]: #pragma omp masked taskloop simd if(cond) firstprivate(fp) reduction(+: rd) linear(lin: step)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#MTLSB+3]]];
#pragma omp masked taskloop simd if(cond) firstprivate(fp) reduction(+:rd) linear(lin: step)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tdpf(int argc)
void tdpf(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TDPF:]]: [B1.{{.+}}]
// CHECK-DAG:  [[#TDPF+1]]: [B1.[[#TDPF+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TDPF+2]]: [B1.[[#TDPF+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TDPF+3]]: [B1.[[#TDPF+7]]]
// CHECK-DAG:  [[#TDPF+4]]: [B1.[[#TDPF+8]]]
// CHECK-DAG:  [[#TDPF+5]]: #pragma omp teams distribute parallel for if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TDPFB:]]];
// CHECK-DAG:  [[#TDPF+6]]: cond
// CHECK-DAG:  [[#TDPF+7]]: fp
// CHECK-DAG:  [[#TDPF+8]]: rd
// CHECK-DAG:  [[#TDPF+9]]: argc
// CHECK-DAG:  [[#TDPF+10]]: x
// CHECK-DAG:  [[#TDPF+11]]: #pragma omp target
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TDPFB-3]]: x
// CHECK-DAG:  [[#TDPFB-2]]: [B3.[[#TDPFB-3]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TDPFB-1]]: argc
// CHECK-DAG:  [[#TDPFB]]: [B3.[[#TDPFB-1]]] = [B3.[[#TDPFB-2]]]
#pragma omp target
#pragma omp teams distribute parallel for if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tdpfs(int argc)
void tdpfs(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TDPFS:]]: [B1.{{.+}}]
// CHECK-DAG:  [[#TDPFS+1]]: [B1.[[#TDPFS+6]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TDPFS+2]]: [B1.[[#TDPFS+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TDPFS+3]]: [B1.[[#TDPFS+7]]]
// CHECK-DAG:  [[#TDPFS+4]]: [B1.[[#TDPFS+8]]]
// CHECK-DAG:  [[#TDPFS+5]]: #pragma omp teams distribute parallel for simd if(cond) firstprivate(fp) reduction(+: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TDPFSB:]]];
// CHECK-DAG:  [[#TDPFS+6]]: cond
// CHECK-DAG:  [[#TDPFS+7]]: fp
// CHECK-DAG:  [[#TDPFS+8]]: rd
// CHECK-DAG:  [[#TDPFS+9]]: argc
// CHECK-DAG:  [[#TDPFS+10]]: x
// CHECK-DAG:  [[#TDPFS+11]]: #pragma omp target
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TDPFSB-3]]: x
// CHECK-DAG:  [[#TDPFSB-2]]: [B3.[[#TDPFSB-3]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TDPFSB-1]]: argc
// CHECK-DAG:  [[#TDPFSB]]: [B3.[[#TDPFSB-1]]] = [B3.[[#TDPFSB-2]]]
#pragma omp target
#pragma omp teams distribute parallel for simd if(cond) firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void tds(int argc)
void tds(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TDS:]]: #pragma omp teams distribute simd firstprivate(fp) reduction(+: rd)
// CHECK-DAG:  [[#TDS-2]]: [B1.[[#TDS+1]]]
// CHECK-DAG:  [[#TDS-1]]: [B1.[[#TDS+2]]]
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TDSB:]]];
// CHECK-DAG:  [[#TDS+1]]: fp
// CHECK-DAG:  [[#TDS+2]]: rd
// CHECK-DAG:  [[#TDS+3]]: argc
// CHECK-DAG:  [[#TDS+4]]: x
// CHECK-DAG:  [[#TDS+5]]: #pragma omp target
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TDSB-3]]: x
// CHECK-DAG:  [[#TDSB-2]]: [B3.[[#TDSB-3]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TDSB-1]]: argc
// CHECK-DAG:  [[#TDSB]]: [B3.[[#TDSB-1]]] = [B3.[[#TDSB-2]]]
#pragma omp target
#pragma omp teams distribute simd firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void teamsloop(int argc)
void teamsloop(int argc) {
  int x, cond, fp, rd, lin, step, map;
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TDS:]]: #pragma omp teams loop firstprivate(fp) reduction(+: rd)
// CHECK-DAG:  [[#TDS-2]]: [B1.[[#TDS+1]]]
// CHECK-DAG:  [[#TDS-1]]: [B1.[[#TDS+2]]]
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TDSB:]]];
// CHECK-DAG:  [[#TDS+1]]: fp
// CHECK-DAG:  [[#TDS+2]]: rd
// CHECK-DAG:  [[#TDS+3]]: argc
// CHECK-DAG:  [[#TDS+4]]: x
// CHECK-DAG:  [[#TDS+5]]: #pragma omp target
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TDSB-3]]: x
// CHECK-DAG:  [[#TDSB-2]]: [B3.[[#TDSB-3]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TDSB-1]]: argc
// CHECK-DAG:  [[#TDSB]]: [B3.[[#TDSB-1]]] = [B3.[[#TDSB-2]]]
#pragma omp target
#pragma omp teams loop firstprivate(fp) reduction(+:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void targetteamsloop(int argc)
void targetteamsloop(int argc) {
  int x, cond, fp, rd, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TTDB:]]: x
// CHECK-DAG:  [[#TTDB+1]]: [B3.[[#TTDB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDB+2]]: argc
// CHECK-DAG:  [[#TTDB+3]]: [B3.[[#TTDB+2]]] = [B3.[[#TTDB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TTD:]]: cond
// CHECK-DAG:  [[#TTD+1]]: [B1.[[#TTD]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTD+2]]: [B1.[[#TTD+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TTD+3]]: fp
// CHECK-DAG:  [[#TTD+4]]: rd
// CHECK-DAG:  [[#TTD+5]]: [B3.[[#TTDB+2]]]
// CHECK-DAG:  [[#TTD+6]]: [B3.[[#TTDB]]]
// CHECK-DAG:  [[#TTD+7]]: #pragma omp target teams loop if(cond) firstprivate(fp) reduction(+: rd) map(alloc: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TTDB+3]]];
#pragma omp target teams loop if(cond) firstprivate(fp) reduction(+:rd) map(alloc:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void parallelloop(int argc)
void parallelloop(int argc) {
  int x, cond, fp, rd;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#PFB:]]: x
// CHECK-DAG:  [[#PFB+1]]: [B3.[[#PFB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PFB+2]]: argc
// CHECK-DAG:  [[#PFB+3]]: [B3.[[#PFB+2]]] = [B3.[[#PFB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#PF:]]: cond
// CHECK-DAG:  [[#PF+1]]: [B1.[[#PF]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#PF+2]]: [B1.[[#PF+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#PF+3]]: fp
// CHECK-DAG:  [[#PF+4]]: rd
// CHECK-DAG:  [[#PF+5]]: #pragma omp parallel loop if(cond) firstprivate(fp) reduction(&: rd)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#PFB+3]]];
#pragma omp parallel loop if(cond) firstprivate(fp) reduction(&:rd)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void targetparallelloop(int argc)
void targetparallelloop(int argc) {
  int x, cond, fp, rd, map;
// CHECK-DAG:   [B3]
// CHECK-DAG:  [[#TTDB:]]: x
// CHECK-DAG:  [[#TTDB+1]]: [B3.[[#TTDB]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTDB+2]]: argc
// CHECK-DAG:  [[#TTDB+3]]: [B3.[[#TTDB+2]]] = [B3.[[#TTDB+1]]]
// CHECK-DAG:   [B1]
// CHECK-DAG:  [[#TTD:]]: cond
// CHECK-DAG:  [[#TTD+1]]: [B1.[[#TTD]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-DAG:  [[#TTD+2]]: [B1.[[#TTD+1]]] (ImplicitCastExpr, IntegralToBoolean, _Bool)
// CHECK-DAG:  [[#TTD+3]]: fp
// CHECK-DAG:  [[#TTD+4]]: rd
// CHECK-DAG:  [[#TTD+5]]: [B3.[[#TTDB+2]]]
// CHECK-DAG:  [[#TTD+6]]: [B3.[[#TTDB]]]
// CHECK-DAG:  [[#TTD+7]]: #pragma omp target parallel loop if(cond) firstprivate(fp) reduction(+: rd) map(alloc: map)
// CHECK-DAG:    for (int i = 0;
// CHECK-DAG:        [B3.[[#TTDB+3]]];
#pragma omp target parallel loop if(cond) firstprivate(fp) reduction(+:rd) map(alloc:map)
  for (int i = 0; i < 10; ++i)
    argc = x;
}

// CHECK-LABEL:  void parallelmasked(int argc)
void parallelmasked(int argc) {
  int x = 0;
// CHECK:        [B1]
// CHECK-NEXT:   1: 0
// CHECK-NEXT:   2: int x = 0;
// CHECK-NEXT:   [[#MASKED:]]: x
// CHECK-NEXT:   [[#MASKED+1]]: [B1.[[#MASKED]]] (ImplicitCastExpr, LValueToRValue, int)
// CHECK-NEXT:   [[#MASKED+2]]: argc
// CHECK-NEXT:   [[#MASKED+3]]: [B1.[[#MASKED+2]]] = [B1.[[#MASKED+1]]]
// CHECK-NEXT:   [[#MASKED+4]]: #pragma omp parallel masked
#pragma omp parallel masked
  argc = x;
}
#endif  // _OPENMP
