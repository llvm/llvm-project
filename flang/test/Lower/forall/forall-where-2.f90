! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s


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

! CHECK:  func @_QPtest_nested_forall_where_with_temp_in_mask({{.*}}) {
! CHECK:   %[[tempResultBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = ".result"}
           ! Where condition pre-evaluation 
! CHECK:   fir.do_loop {{.*}} {
! CHECK:      fir.do_loop {{.*}} {
                ! Evaluation of mask for iteration (i,j) into ragged array temp 
! CHECK:        %[[tempResult:.*]] = fir.call @_QPtemp_foo
! CHECK:        fir.save_result %[[tempResult]] to %[[tempResultBox]] : !fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:        fir.if {{.*}} {
! CHECK:          @_FortranARaggedArrayAllocate
! CHECK:        }
! CHECK:        fir.do_loop {{.*}} {
                  ! store into ragged array temp element
! CHECK:        }
! CHECK:        %[[box:.*]] = fir.load %[[tempResultBox]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:        %[[tempAddr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
                ! local temps that were generated during the evaluation are cleaned-up after the value were stored
                ! into the ragged array temp.
! CHECK:        fir.freemem %[[tempAddr]] : !fir.heap<!fir.array<?xf32>>
! CHECK:      }
! CHECK:    }
            ! Where assignment
! CHECK:    fir.do_loop {{.*}} {
! CHECK:      fir.do_loop {{.*}} {
                ! Array assignment at iteration (i, j)
! CHECK:        fir.do_loop {{.*}} {
! CHECK:          fir.if {{.*}} {  
! CHECK:            arith.divf
! CHECK:          } else {
! CHECK:          }
! CHECK:        }
! CHECK:      }
! CHECK:    }
            ! Elsewhere assignment
! CHECK:    fir.do_loop {{.*}} {
! CHECK:      fir.do_loop {{.*}} {
                ! Array assignment at iteration (i, j)
! CHECK:        fir.do_loop {{.*}} {
! CHECK:          fir.if {{.*}} {  
! CHECK:          } else {
! CHECK:            arith.negf
! CHECK:          }
! CHECK:        }
! CHECK:      }
! CHECK:    }
            ! Ragged array clean-up
! CHECK:    fir.call @_FortranARaggedArrayDeallocate
! CHECK:  }
