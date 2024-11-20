! This test checks lowering of OpenMP declare mapper Directive.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/declare-mapper-1.f90 -o - | FileCheck %t/declare-mapper-1.f90
! RUN  %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/declare-mapper-2.f90 -o - | FileCheck %t/declare-mapper-2.f90

!--- declare-mapper-1.f90
subroutine mapper
   implicit none
   type my_type
      integer, pointer :: my_buffer
      integer :: my_buffer_size
   end type
   !CHECK: %[[MY_VAR:.*]] = fir.alloca ![[VAR_TYPE:.*]] {bindc_name = "my_var"}
   !CHECK: %[[MAP_INFO:.*]] = omp.map.info var_ptr(%[[MY_VAR]] : !fir.ref<![[VAR_TYPE]]>, ![[VAR_TYPE]]) map_clauses(tofrom) capture(ByRef) -> !fir.ref<![[VAR_TYPE]]> {name = "my_var"}
   !CHECK: omp.declare_mapper @my_mapper : %[[MY_VAR]] : !fir.ref<![[VAR_TYPE]]> : ![[VAR_TYPE]] map_entries(%[[MAP_INFO]] : !fir.ref<![[VAR_TYPE]]>)
   !$omp DECLARE MAPPER(my_mapper : my_type :: my_var) map(tofrom : my_var)
end subroutine

!--- declare-mapper-2.f90
subroutine mapper_default
   implicit none
   type my_type
      integer, pointer :: my_buffer
      integer :: my_buffer_size
   end type
   !CHECK: %[[MY_VAR:.*]] = fir.alloca ![[VAR_TYPE:.*]] {bindc_name = "my_var"}
   !CHECK: %[[MAP_INFO:.*]] = omp.map.info var_ptr(%[[MY_VAR]] : !fir.ref<![[VAR_TYPE]]>, ![[VAR_TYPE]]) map_clauses(tofrom) capture(ByRef) -> !fir.ref<![[VAR_TYPE]]> {name = "my_var"}
   !CHECK: omp.declare_mapper @default_my_type : %[[MY_VAR]] : !fir.ref<![[VAR_TYPE]]> : ![[VAR_TYPE]] map_entries(%[[MAP_INFO]] : !fir.ref<![[VAR_TYPE]]>)
   !$omp DECLARE MAPPER(my_type :: my_var) map(tofrom : my_var)
end subroutine
