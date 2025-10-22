! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine depend
   type :: my_struct
       integer :: my_component(10)
   end type

   type(my_struct) :: my_var

   !$omp task depend(in:my_var%my_component)
   !$omp end task
end subroutine depend

! CHECK: %[[VAR_ALLOC:.*]] = fir.alloca !fir.type<{{.*}}my_struct{{.*}}> {bindc_name = "my_var", {{.*}}}
! CHECK: %[[VAR_DECL:.*]]:2 = hlfir.declare %[[VAR_ALLOC]]

! CHECK: %[[COMP_SELECTOR:.*]] = hlfir.designate %[[VAR_DECL]]#0{"my_component"}

! CHECK: omp.task depend(taskdependin -> %[[COMP_SELECTOR]] : {{.*}}) {
! CHECK:   omp.terminator
! CHECK: }
