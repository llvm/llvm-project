!RUN: rm -rf %t && mkdir %t && cd %t && \
!RUN:   bbc %s -fopenacc -emit-hlfir -o - \
!RUN:   | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" \
!RUN:   | FileCheck %s --check-prefix=CHECKHLFIR

!RUN: rm -rf %t && mkdir %t && cd %t && \
!RUN:   bbc %s -fopenacc -emit-hlfir -o - \
!RUN:   | fir-opt --pass-pipeline="builtin.module(cse,acc-initialize-fir-analyses,acc-implicit-data)" \
!RUN:   | FileCheck %s --check-prefix=CHECKCSE

!RUN: rm -rf %t && mkdir %t && cd %t && \
!RUN:   bbc %s -fopenacc -emit-fir -o - \
!RUN:   | fir-opt --pass-pipeline="builtin.module(cse,acc-initialize-fir-analyses,acc-implicit-data)" \
!RUN:   | FileCheck %s --check-prefix=CHECKCSE

! This test uses bbc to generate both HLFIR and FIR for this test. The intent is
! that it is exercising the acc implicit data pipeline and ensures that
! correct clauses are generated. It also runs CSE which eliminates redundant
! interior pointer computations (and thus different live-ins are found).

program main
  type aggr
    real :: field
  end type
  type nested
    type(aggr) :: outer
  end type
  type(aggr) :: aggrvar
  type(nested) :: nestaggrvar
  real :: scalarvar
  real :: arrayvar(10)
  complex :: scalarcomp

  aggrvar%field = 1
  scalarvar = aggrvar%field
  nestaggrvar%outer%field = scalarvar
  scalarcomp = scalarvar
  arrayvar = real(scalarcomp)
  arrayvar(2) = aggrvar%field

  !$acc kernels
  arrayvar = aggrvar%field + scalarvar + nestaggrvar%outer%field + real(scalarcomp) + arrayvar(2)
  !$acc end kernels

  !$acc parallel
  arrayvar = aggrvar%field + scalarvar + nestaggrvar%outer%field + real(scalarcomp) + arrayvar(2)
  !$acc end parallel
end program

!CHECKHLFIR-LABEL: @_QQmain
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) dataClause(acc_copy) implicit(true) name("aggrvar") -> !fir.ref<!fir.type<_QFTaggr{field:f32}>>
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) dataClause(acc_copy) implicit(true) name("arrayvar") -> !fir.ref<!fir.array<10xf32>>
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.type<_QFTnested{outer:!fir.type<_QFTaggr{field:f32}>}>>) dataClause(acc_copy) implicit(true) name("nestaggrvar") -> !fir.ref<!fir.type<_QFTnested{outer:!fir.type<_QFTaggr{field:f32}>}>>
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<complex<f32>>) dataClause(acc_copy) implicit(true) name("scalarcomp") -> !fir.ref<complex<f32>>
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("scalarvar") -> !fir.ref<f32>
!CHECKHLFIR: acc.kernels
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}}  : !fir.ref<!fir.type<_QFTaggr{field:f32}>>) dataClause(acc_copy) implicit(true) name("aggrvar") -> !fir.ref<!fir.type<_QFTaggr{field:f32}>>
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}}  : !fir.ref<!fir.array<10xf32>>) dataClause(acc_copy) implicit(true) name("arrayvar") -> !fir.ref<!fir.array<10xf32>>
!CHECKHLFIR-DAG: acc.copyin varPtr(%{{.*}}  : !fir.ref<!fir.type<_QFTnested{outer:!fir.type<_QFTaggr{field:f32}>}>>) dataClause(acc_copy) implicit(true) name("nestaggrvar") -> !fir.ref<!fir.type<_QFTnested{outer:!fir.type<_QFTaggr{field:f32}>}>>
!CHECKHLFIR-DAG: acc.firstprivate varPtr(%{{.*}}  : !fir.ref<complex<f32>>) recipe({{.*}}) implicit(true) name("scalarcomp") -> !fir.ref<complex<f32>>
!CHECKHLFIR-DAG: acc.firstprivate varPtr(%{{.*}}  : !fir.ref<f32>) recipe({{.*}}) implicit(true) name("scalarvar") -> !fir.ref<f32>
!CHECKHLFIR: acc.parallel

!CHECKCSE-LABEL: @_QQmain
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) dataClause(acc_copy) implicit(true) name("arrayvar") -> !fir.ref<!fir.array<10xf32>>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<complex<f32>>) dataClause(acc_copy) implicit(true) name("scalarcomp") -> !fir.ref<complex<f32>>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("scalarvar") -> !fir.ref<f32>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("aggrvar%field") -> !fir.ref<f32>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("nestaggrvar%outer%field") -> !fir.ref<f32>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("arrayvar(2)") -> !fir.ref<f32>
!CHECKCSE: acc.kernels
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) dataClause(acc_copy) implicit(true) name("arrayvar") -> !fir.ref<!fir.array<10xf32>>
!CHECKCSE-DAG: acc.firstprivate varPtr(%{{.*}} : !fir.ref<complex<f32>>) recipe({{.*}}) implicit(true) name("scalarcomp") -> !fir.ref<complex<f32>>
!CHECKCSE-DAG: acc.firstprivate varPtr(%{{.*}} : !fir.ref<f32>) recipe({{.*}}) implicit(true) name("scalarvar") -> !fir.ref<f32>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("aggrvar%field") -> !fir.ref<f32>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("nestaggrvar%outer%field") -> !fir.ref<f32>
!CHECKCSE-DAG: acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) dataClause(acc_copy) implicit(true) name("arrayvar(2)") -> !fir.ref<f32>
!CHECKCSE: acc.parallel

