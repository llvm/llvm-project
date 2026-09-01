! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -module-dir %t %t/mapper.f90 -o /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -J %t %t/consumer.f90 -o - | FileCheck %s

!--- mapper.f90
module iterator_mapper_mod
  type :: t
    integer :: a(10)
  end type
  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): v%a(i))
end module

!--- consumer.f90
subroutine use_iterator_mapper_mod
  use iterator_mapper_mod
  integer :: k
  k = 1
end subroutine

! CHECK-LABEL: omp.declare_mapper @_QQMiterator_mapper_modm
! CHECK: %[[ITER:.*]] = omp.iterator
! CHECK: %[[MAP:.*]] = omp.map.info
! CHECK: omp.yield(%[[MAP]] : !llvm.ptr)
! CHECK: omp.declare_mapper.info map_iterated(%[[ITER]]
