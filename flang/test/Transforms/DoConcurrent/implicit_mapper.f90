! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

module record_with_alloc_mod
  implicit none
  public :: record_with_alloc

  type record_with_alloc
    real, allocatable :: values_(:)
  end type
end module record_with_alloc_mod

subroutine random_inputs()
  use record_with_alloc_mod, only : record_with_alloc
  implicit none
  type(record_with_alloc) :: inputs(2)
  integer :: i

  do concurrent(i=1:10)
    inputs(1)%values_ = [1,2,3,4]
  end do
end subroutine

! CHECK: omp.declare_mapper @[[MAPPER_NAME:.*record_with_alloc_omp_default_mapper]] : !fir.type<{{.*}}record_with_alloc{{.*}}>

! CHECK: func.func @{{.*}}random_inputs()
! CHECK:   %[[ARR_DECL:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "{{.*}}inputs"}
! CHECK:   omp.map.info var_ptr(%[[ARR_DECL]]#1 : {{.*}}) {{.*}} mapper(@[[MAPPER_NAME]])
