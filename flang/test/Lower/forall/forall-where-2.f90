! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test a FORALL construct with a nested WHERE construct where the mask
! contains temporary array expressions.

subroutine test_nested_forall_where_with_temp_in_mask(a,b)
  interface
    function temp_foo(i, j)
      integer :: i, j
      real, allocatable :: temp_foo(:)
    end function
  end interface
  type t
     real data(100)
  end type t
  type(t) :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2))
     where (b(j,i)%data > temp_foo(i, j))
        a(i,j)%data = b(j,i)%data / 3.14
     elsewhere
        a(i,j)%data = -b(j,i)%data
     end where
  end forall
end subroutine

! CHECK-LABEL:  func.func @_QPtest_nested_forall_where_with_temp_in_mask({{.*}}) {
! CHECK:   %[[tempResultBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = ".result"}
! CHECK:   hlfir.forall
! CHECK:   (%[[arg2:.*]]: i32) {
! CHECK:     %[[i:.*]] = hlfir.forall_index "i" %[[arg2]] : (i32) -> !fir.ref<i32>
! CHECK:     hlfir.forall
! CHECK:     (%[[arg3:.*]]: i32) {
! CHECK:       %[[j:.*]] = hlfir.forall_index "j" %[[arg3]] : (i32) -> !fir.ref<i32>
! CHECK:       hlfir.where {
! CHECK:         %[[tempResult:.*]] = fir.call @_QPtemp_foo(%[[i]], %[[j]])
! CHECK:         fir.save_result %[[tempResult]] to {{.*}}
! CHECK:         %[[mask:.*]] = hlfir.elemental {{.*}} {
! CHECK:           arith.cmpf ogt, {{.*}}
! CHECK:         }
! CHECK:         hlfir.yield %[[mask]]
! CHECK:       } do {
! CHECK:         hlfir.region_assign {
! CHECK:           %[[res:.*]] = hlfir.elemental {{.*}} {
! CHECK:             arith.divf {{.*}}
! CHECK:           }
! CHECK:           hlfir.yield %[[res]]
! CHECK:         } to {
! CHECK:         }
! CHECK:       hlfir.elsewhere do {
! CHECK:         hlfir.region_assign {
! CHECK:           %[[res:.*]] = hlfir.elemental {{.*}} {
! CHECK:             arith.negf {{.*}}
! CHECK:           }
! CHECK:           hlfir.yield %[[res]]
! CHECK:         } to {
! CHECK:         }
! CHECK:       }
! CHECK:     }
! CHECK:   }
! CHECK: }
