! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s
module mod
  type t
     integer::i=0
  end type t
!$omp declare reduction (+:t:omp_out%i=omp_out%i+omp_in%i) &
!$omp         initializer(omp_priv%i=0)
end module mod

!CHECK: Module scope: mod
!CHECK: op.+, PUBLIC: UserReductionDetails TYPE(t)
!CHECK: t, PUBLIC: DerivedType components: i

program main
  use mod
  integer::i
  type(t)::x1
  x1%i=0
!$omp parallel do reduction(+:x1)
  do i=1,10
     x1%i=x1%i+1
  end do
!$omp end parallel do
  print *,'pass'
end program main

!CHECK: MainProgram scope: MAIN
!CHECK: op.+: Use from op.+ in mod
!CHECK: t: Use from t in mod
!CHECK: x1 size=4 offset=4: ObjectEntity type: TYPE(t)
