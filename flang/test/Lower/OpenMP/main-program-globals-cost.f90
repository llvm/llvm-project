! Check reuse of storage-selection and host-binding analysis results. Trace
! checks require assertions; the IR checks also run without the asserts feature.
! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %if asserts %{ bbc -fopenmp -pft-test -debug-only=flang-pft %t/saved.f90 2>&1 | FileCheck %s --check-prefix=OFF %}
! RUN: %flang_fc1 -fopenmp -fno-automatic -emit-hlfir %if asserts %{ -mllvm -debug-only=flang-pft %} %t/nested.f90 -o %t.auto > %t.auto-trace 2>&1
! RUN: %if asserts %{ FileCheck %s --check-prefix=OFF --allow-empty < %t.auto-trace %}
! RUN: FileCheck %s --check-prefix=SAVED < %t.auto
! RUN: %if asserts %{ bbc -fopenmp -pft-test -debug-only=flang-pft %t/pruned.f90 2>&1 | FileCheck %s --check-prefix=PRUNED %}
! RUN: %if asserts %{ bbc -fopenmp -pft-test -debug-only=flang-pft %t/nested.f90 2>&1 | FileCheck %s --check-prefix=EXPR %}
! RUN: %if asserts %{ bbc -fopenmp -fno-openmp-main-program-globals -pft-test -debug-only=flang-pft %t/nested.f90 2>&1 | FileCheck %s --check-prefix=OFF %}
! RUN: %flang_fc1 -fopenmp -emit-hlfir -module-dir %t %if asserts %{ -mllvm -debug-only=flang-lower-bridge,flang-lower-variable,flang-pft,flang-host-assoc %} %t/owner.f90 -o %t.owner > %t.trace 2>&1
! RUN: FileCheck %s --check-prefix=IR < %t.owner
! RUN: FileCheck %s --check-prefix=UNUSED < %t.owner
! RUN: %if asserts %{ FileCheck %s --check-prefix=STORE < %t.trace %}
! RUN: %if asserts %{ FileCheck %s --check-prefix=CACHE < %t.trace %}
! RUN: %if asserts %{ FileCheck %s --check-prefix=ROOTS < %t.trace %}
! RUN: %if asserts %{ FileCheck %s --check-prefix=TAILS < %t.trace %}
! RUN: %flang_fc1 -fopenmp -fsave-main-program -emit-hlfir %if asserts %{ -mllvm -debug-only=flang-pft %} %t/nested.f90 -o %t.saved > %t.save-trace 2>&1
! RUN: %if asserts %{ FileCheck %s --check-prefix=OFF --allow-empty < %t.save-trace %}
! RUN: FileCheck %s --check-prefix=SAVED < %t.saved
! SAVED-DAG: fir.global internal @_QFEsink : i32
! SAVED-DAG: fir.global internal @_QFEx1 : i32
! EXPR-COUNT-18: main-global symbol
! EXPR-NOT: main-global symbol
! EXPR: Program
! EXPR-NOT: main-global symbol
! OFF-NOT: {{^main-global }}
! CACHE-NOT: getScopeVariableList of [sub]program|block scope {{.*}} P{{$}}
! CACHE-NOT: getScopeVariableList of main program scope
! CACHE: getScopeVariableList of main program scope
! CACHE-NOT: getScopeVariableList of main program scope
! CACHE-NOT: getScopeVariableList of [sub]program|block scope {{.*}} P{{$}}
! STORE: [bridge - startNewFunction] worker
! STORE-NOT: instantiateVariable: interval{{.*}} c,
! STORE-DAG: instantiateVariable: interval{{.*}} a, global
! STORE-DAG: instantiateVariable: interval{{.*}} alias, global
! STORE-NOT: instantiateVariable: interval
! STORE: [bridge - endNewFunction
! STORE: [bridge - startNewFunction] other
! STORE-NOT: instantiateVariable: interval{{.*}} c,
! STORE-DAG: instantiateVariable: interval{{.*}} a, global
! STORE-DAG: instantiateVariable: interval{{.*}} alias, global
! STORE-NOT: instantiateVariable: interval
! STORE: [bridge - endNewFunction
! IR-LABEL: func.func private @_QFPworker()
! IR-DAG: fir.address_of(@_QFEalias)
! IR-DAG: fir.address_of(@_QFEx)
! IR: omp.parallel
! IR: return
! IR-LABEL: func.func private @_QFPother()
! IR-DAG: %[[MODULE:[^ ]+]] = fir.address_of(@_QMneededEa)
! IR-DAG: %[[COMMON:[^ ]+]] = fir.address_of(@block_)
! IR-DAG: %[[USED:[^ ]+]]:2 = hlfir.declare {{.*}} storage(%[[MODULE]][0]) uniq_name("_QMneededEb")
! IR-DAG: %[[SHARED:[^ ]+]]:2 = hlfir.declare {{.*}} storage(%[[COMMON]][0]) uniq_name("_QFEshared")
! IR: omp.parallel
! IR-DAG: fir.load %[[USED]]#0 :
! IR-DAG: fir.load %[[SHARED]]#0 :
! IR: return
! UNUSED-LABEL: func.func private @_QFPworker()
! UNUSED-NOT: fir.address_of(@_QM{{(neededEc|tail0E)}}
! UNUSED: return
! UNUSED-LABEL: func.func private @_QFPother()
!--- nested.f90
! Visit nested operands once; SAVE policies and the opt-out bypass selection.
program p
integer :: x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,sink
!$omp parallel
if (x1>0) sink=x1+(x2+(x3+(x4+(x5+(x6+(x7+(x8+( &
     x9+(x10+(x11+(x12+(x13+(x14+(x15+x16))))))))))))))
!$omp end parallel
end program
!--- owner.f90
! Reuse the MAIN variable list and shared USE tails. Bind only the required
! globals and EQUIVALENCE/COMMON stores in each internal procedure.
module needed
  integer :: a,b,c,d; equivalence(a,b),(c,d)
end module
module tail0; integer :: v; end module; module tail1; use tail0, only: v; end module
module tail2; use tail1, only: v; end module; module tail3; use tail2, only: v; end module
program p
  use needed, only: unused=>a,used=>b
  use tail3, only: a1=>v,a2=>v,a3=>v,a4=>v
  integer,save :: backing(2),alias; equivalence(backing(1),alias)
  integer :: x,shared; common /block/ shared
  alias=3; x=7; used=11; shared=13; call worker(); call other()
contains
  subroutine worker()
!$omp parallel num_threads(1)
    print *,x,alias
!$omp end parallel
  end subroutine; subroutine other()
!$omp parallel num_threads(1)
    print *,x,alias,used,shared
!$omp end parallel
  end subroutine; end program
! Four MAIN entities, one module object and six OpenMP proxies are resolved once.
! ROOTS-COUNT-11: main-global association
! ROOTS-NOT: main-global association
! Four renames share three USE edges; two needed aliases add two links.
! TAILS-COUNT-9: main-global binding association
! TAILS-NOT: main-global binding association
! PRUNED-NOT: main-global reference: image
! PRUNED: main-global reference: direct
! PRUNED-NOT: main-global reference: image
! PRUNED: Program
! PRUNED-NOT: main-global reference: image
!--- pruned.f90
! Do not select the coarray image selector; direct remains a candidate.
program p; type t(k)
    integer,kind :: k=4; integer :: payload=1
  end type; type(t),parameter :: c=t()
  integer :: direct,image,co[*]
!$omp parallel
  print *,direct,co[image],c
!$omp end parallel
end program
!--- saved.f90
! A bare SAVE makes the new storage-selection analysis unnecessary.
program p
  integer :: x
  save
!$omp parallel
  x=1
!$omp end parallel
end program
