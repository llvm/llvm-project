! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s --check-prefix=ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fimplicit-default-mapper-allocatable-members %s -o - | FileCheck %s --check-prefix=ON
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fno-implicit-default-mapper-allocatable-members %s -o - | FileCheck %s --check-prefix=OFF

! Check the -f[no-]implicit-default-mapper-allocatable-members switch for
! explicit maps that synthesize an implicit default mapper for a derived
! type containing an allocatable.
!
! With the default/on setting, compiler-generated implicit default mappers for
! derived types include maps for allocatable components. With the off setting,
! those component maps are suppressed.

program explicit_map
  implicit none

  type :: t
    integer :: scalar
    integer, allocatable :: a(:)
  end type

  type(t) :: x

  !$omp target map(tofrom: x)
  !$omp end target
end program

! ON: omp.declare_mapper @[[MAPPER:_QQFt_omp_default_mapper]]
! ON: %[[COORD:.*]] = fir.coordinate_of %{{.*}}#0, a
! ON: %[[PTEE:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(implicit, tofrom, ref_ptee)
! ON: %[[ATTACH:.*]] = omp.map.info var_ptr(%[[COORD]] {{.*}}) map_clauses(attach, ref_ptee)
! ON: %[[PARENT:.*]] = omp.map.info {{.*}}members(%[[PTEE]] : [1]
! ON: omp.declare_mapper.info map_entries(%[[PARENT]], %[[PTEE]], %[[ATTACH]]
! ON: omp.map.info {{.*}}map_clauses(tofrom) capture(ByRef) mapper(@[[MAPPER]]) name("x")
! ON: omp.target

! OFF-NOT: omp.declare_mapper
! OFF: %[[MAP:.*]] = omp.map.info var_ptr(%{{.*}}) map_clauses(tofrom) capture(ByRef) name("x")
! OFF-NOT: mapper(
! OFF: omp.target {{.*}}map_entries(%[[MAP]] ->
