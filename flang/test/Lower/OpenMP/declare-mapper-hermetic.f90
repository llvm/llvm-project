! Hermetic module files embed the modules a module uses into its own .mod. A
! consumer that materializes an imported declare mapper owned by such an embedded
! module dereferences parse-tree pointers (MapperDetails) into it, so the embedded
! module's parse tree must stay live past module-file reading. map_wrap re-exports
! map_base (which owns the mapper) and is compiled hermetically; map_base.mod is
! then removed so the consumer can only reach the mapper through the embedding.

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=50 map_base.f90
! RUN: %flang_fc1 -fhermetic-module-files -fsyntax-only -fopenmp -fopenmp-version=50 map_wrap.f90
! RUN: rm map_base.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 map.use.f90 -o - | FileCheck map.use.f90

!--- map_base.f90
module map_base
  type :: t
    integer :: x = 0
  end type
  !$omp declare mapper(mymapper : t :: v) map(v%x)
end module

!--- map_wrap.f90
module map_wrap
  use map_base
end module

!--- map.use.f90
! CHECK: omp.declare_mapper @[[MAPPER:_QQMmap_base[A-Za-z0-9_.]*mymapper]]
! CHECK: mapper(@[[MAPPER]])
program main
  use map_wrap
  type(t) :: obj
  obj%x = 0
  !$omp target map(mapper(mymapper), tofrom: obj)
  obj%x = obj%x + 1
  !$omp end target
  print *, obj%x
end program