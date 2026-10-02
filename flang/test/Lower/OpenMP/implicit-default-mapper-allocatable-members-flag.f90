! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/implicit.f90 -o - | FileCheck %s --check-prefix=IMPLICIT-ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fimplicit-default-mapper-allocatable-members %t/implicit.f90 -o - | FileCheck %s --check-prefix=IMPLICIT-ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fno-implicit-default-mapper-allocatable-members %t/implicit.f90 -o - | FileCheck %s --check-prefix=IMPLICIT-OFF
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/explicit.f90 -o - | FileCheck %s --check-prefix=EXPLICIT-ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fimplicit-default-mapper-allocatable-members %t/explicit.f90 -o - | FileCheck %s --check-prefix=EXPLICIT-ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fno-implicit-default-mapper-allocatable-members %t/explicit.f90 -o - | FileCheck %s --check-prefix=EXPLICIT-OFF

! Check the -f[no-]implicit-default-mapper-allocatable-members switch.
!
! With the default/on setting, compiler-generated implicit default mappers for
! derived types include maps for allocatable and nested-record components.
! With the off setting, those component maps are suppressed. Cover both:
!   1. an implicit target-region capture; and
!   2. an explicit whole-object map that would otherwise synthesize a default
!      mapper for the derived type.

!--- implicit.f90
program implicit_capture
  implicit none

  type :: t
    integer, allocatable :: a(:)
  end type

  type(t) :: x

  allocate(x%a(1))

  !$omp target
    x%a(1) = 1
  !$omp end target
end program

!--- explicit.f90
program explicit_whole_object_map
  implicit none

  type :: t
    integer :: scalar
    integer, allocatable :: a(:)
  end type

  type(t) :: x

  allocate(x%a(1))
  x%scalar = 1

  !$omp target map(tofrom: x)
    x%scalar = x%scalar + 1
    x%a(1) = x%a(1) + 1
  !$omp end target
end program

! IMPLICIT-ON: omp.declare_mapper @[[IM_MAPPER:_QQFt_omp_default_mapper]]
! IMPLICIT-ON: %[[COORD:.*]] = fir.coordinate_of %{{.*}}#0, a
! IMPLICIT-ON: %[[PTEE:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(implicit, tofrom, ref_ptee)
! IMPLICIT-ON: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(attach, ref_ptee)
! IMPLICIT-ON: %[[PARENT:.*]] = omp.map.info {{.*}}members(%[[PTEE]] : [0]
! IMPLICIT-ON: omp.declare_mapper.info map_entries(%[[PARENT]], %[[PTEE]], %[[ATTACH]]
! IMPLICIT-ON: omp.map.info {{.*}}map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[IM_MAPPER]]) name("x")
! IMPLICIT-ON: omp.target

! IMPLICIT-OFF-NOT: omp.declare_mapper
! IMPLICIT-OFF: %[[IM_MAP:.*]] = omp.map.info var_ptr(%{{.*}}) map_clauses(implicit, tofrom) capture(ByRef) name("x")
! IMPLICIT-OFF-NOT: mapper(
! IMPLICIT-OFF: omp.target {{.*}}map_entries(%[[IM_MAP]] ->

! EXPLICIT-ON: omp.declare_mapper @[[EX_MAPPER:_QQFt_omp_default_mapper]]
! EXPLICIT-ON: %[[COORD:.*]] = fir.coordinate_of %{{.*}}#0, a
! EXPLICIT-ON: %[[PTEE:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(implicit, tofrom, ref_ptee)
! EXPLICIT-ON: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(attach, ref_ptee)
! EXPLICIT-ON: %[[PARENT:.*]] = omp.map.info {{.*}}members(%[[PTEE]] : [1]
! EXPLICIT-ON: omp.declare_mapper.info map_entries(%[[PARENT]], %[[PTEE]], %[[ATTACH]]
! EXPLICIT-ON: omp.map.info {{.*}}map_clauses(tofrom) capture(ByRef) mapper(@[[EX_MAPPER]]) name("x")
! EXPLICIT-ON: omp.target

! EXPLICIT-OFF-NOT: omp.declare_mapper
! EXPLICIT-OFF: %[[EX_MAP:.*]] = omp.map.info var_ptr(%{{.*}}) map_clauses(tofrom) capture(ByRef) name("x")
! EXPLICIT-OFF-NOT: mapper(
! EXPLICIT-OFF: omp.target {{.*}}map_entries(%[[EX_MAP]] ->
