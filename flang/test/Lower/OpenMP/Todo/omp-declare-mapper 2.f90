! This test checks lowering of OpenMP declare mapper Directive.

! RUN: split-file %s %t
! RUN: not %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-1.f90 2>&1 | FileCheck %t/omp-declare-mapper-1.f90
! RUN  not %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 %t/omp-declare-mapper-2.f90 2>&1 | FileCheck %t/omp-declare-mapper-2.f90

!--- omp-declare-mapper-1.f90
subroutine declare_mapper_1
 integer,parameter      :: nvals = 250
 type my_type
   integer              :: num_vals
   integer, allocatable :: values(:)
 end type 

 type my_type2
   type (my_type)        :: my_type_var
   type (my_type)        :: temp
   real,dimension(nvals) :: unmapped
   real,dimension(nvals) :: arr
  end type
  type (my_type2)        :: t
  real                   :: x, y(nvals)
  !$omp declare mapper (my_type :: var) map (var, var%values (1:var%num_vals))
!CHECK: not yet implemented: OpenMPDeclareMapperConstruct
end subroutine declare_mapper_1


!--- omp-declare-mapper-2.f90
subroutine declare_mapper_2
 integer,parameter      :: nvals = 250
 type my_type
   integer              :: num_vals
   integer, allocatable :: values(:)
 end type 

 type my_type2
   type (my_type)        :: my_type_var
   type (my_type)        :: temp
   real,dimension(nvals) :: unmapped
   real,dimension(nvals) :: arr
  end type
  type (my_type2)        :: t
  real                   :: x, y(nvals)
  !$omp declare mapper (my_mapper : my_type2 :: v) map (v%arr, x, y(:)) &
  !$omp&                map (alloc : v%temp)
!CHECK: not yet implemented: OpenMPDeclareMapperConstruct
end subroutine declare_mapper_2
