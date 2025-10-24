!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module mod
contains
   subroutine foo(dt, switch)
      implicit none
      real(4), dimension(:), intent(inout) :: dt
      logical, intent(in) :: switch
      integer :: dim, idx

      if (switch) then
!$omp target teams distribute parallel do
         do idx = 1, 100
            dt(idx) = 20
         end do
      else
!$omp target teams distribute parallel do
         do idx = 1, 100
            dt(idx) = 30
         end do
      end if
   end subroutine foo
end module

! CHECK-LABEL:   func.func @{{.*}}(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "dt"},
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope {{.*}}
! CHECK:           fir.if %{{.*}} {
! CHECK:             %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:             fir.if %[[VAL_2]] {
! CHECK:               fir.store %[[VAL_1]]#1 to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:             }
! CHECK:             %[[VAL_3:.*]] = fir.box_offset %[[VAL_0]] base_addr : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
! CHECK:             %[[VAL_4:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, f32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_3]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>> {name = ""}
! CHECK:             %[[VAL_5:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.box<!fir.array<?xf32>>) map_clauses(implicit, to) capture(ByRef) members(%[[VAL_4]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) -> !fir.ref<!fir.array<?xf32>> {name = "dt"}
! CHECK:             omp.target host_eval({{.*}}) map_entries({{.*}}%[[VAL_5]] -> {{.*}}, %[[VAL_4]] -> {{.*}} : {{.*}}) {
! CHECK:           } else {
! CHECK:             %[[VAL_6:.*]] = fir.is_present %[[VAL_1]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:             fir.if %[[VAL_6]] {
! CHECK:               fir.store %[[VAL_1]]#1 to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:             }
! CHECK:             %[[VAL_7:.*]] = fir.box_offset %[[VAL_0]] base_addr : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
! CHECK:             %[[VAL_8:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, f32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_7]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>> {name = ""}
! CHECK:             %[[VAL_9:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.box<!fir.array<?xf32>>) map_clauses(implicit, to) capture(ByRef) members(%[[VAL_8]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) -> !fir.ref<!fir.array<?xf32>> {name = "dt"}
! CHECK:             omp.target host_eval({{.*}}) map_entries({{.*}}, %[[VAL_9]] ->{{.*}}, %[[VAL_8]] -> {{.*}} : {{.*}}) {
