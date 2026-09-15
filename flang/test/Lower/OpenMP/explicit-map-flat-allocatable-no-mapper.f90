! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=NO-FLAT

! On an explicit map clause, an allocatable array of a derived type whose
! components are all scalars/fixed-size arrays (a "flat" record needing no
! deep copy) must NOT get an implicit default mapper synthesized. A derived
! type that has an allocatable component still requires a mapper, and one
! must still be generated after the fix.

module types_mod
  type flat_ty
    integer :: i
    real    :: r(3)
  end type flat_ty

  type deep_ty
    integer              :: i
    real, allocatable    :: a(:)
  end type deep_ty
end module types_mod

subroutine map_flat()
  use types_mod
  type(flat_ty), allocatable :: arr(:)
  allocate(arr(100))
  !$omp target map(tofrom: arr)
    arr(1)%i = arr(1)%i + 1
  !$omp end target
end subroutine map_flat

subroutine map_deep()
  use types_mod
  type(deep_ty), allocatable :: arr(:)
  allocate(arr(100))
  !$omp target map(tofrom: arr)
    arr(1)%i = arr(1)%i + 1
  !$omp end target
end subroutine map_deep

! Verify no flat_ty mapper is generated anywhere in the program.
! NO-FLAT-NOT: omp.declare_mapper @{{.*}}flat_ty{{.*}}
! NO-FLAT-NOT: mapper(@{{.*}}flat_ty{{.*}})

! Verify we do at least emit and attach the deep_ty mapper.
! CHECK: omp.declare_mapper @{{.*}}deep_ty{{.*}}
! CHECK-LABEL: func.func @_QPmap_deep
! CHECK: omp.map.info {{.*}}mapper(@{{.*}}deep_ty{{.*}})
