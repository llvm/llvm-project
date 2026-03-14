! Test that WHERE mask clean-up occurs at the right time when the
! WHERE contains whole allocatable assignments.
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

module mtest
contains

! CHECK-LABEL: func.func @_QMmtestPfoo(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:       %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "b"}) {
subroutine foo(a, b)
  integer :: a(:)
  integer, allocatable :: b(:)
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
          ! WHERE mask temp allocation
! CHECK:  %[[VAL_9:.*]] = fir.allocmem !fir.array<?x!fir.logical<4>>, %[[VAL_4]]#1 {uniq_name = ".array.expr"}
! CHECK:  %[[VAL_15:.*]] = fir.do_loop {{.*}} {
!           ! WHERE mask element computation
! CHECK:  }
! CHECK:  fir.array_merge_store %{{.*}}, %[[VAL_15]] to %[[VAL_9]] : !fir.array<?x!fir.logical<4>>, !fir.array<?x!fir.logical<4>>, !fir.heap<!fir.array<?x!fir.logical<4>>>

          ! First assignment to a whole allocatable (in WHERE)
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.if {{.*}} {
            ! assignment into new storage (`b` allocated with bad shape)
! CHECK:      fir.allocmem
! CHECK:      fir.do_loop {{.*}} {
! CHECK:        fir.array_coor %[[VAL_9]]
! CHECK:        fir.if %{{.*}} {
                  ! WHERE
! CHECK:          fir.array_update {{.*}}
! CHECK:        } else {
! CHECK:        }
! CHECK:      }
! CHECK:    } else {
              ! assignment into old storage (`b` allocated with the same shape)
! CHECK:      fir.do_loop {{.*}} {
! CHECK:        fir.array_coor %[[VAL_9]]
! CHECK:        fir.if %{{.*}} {
                  ! WHERE
! CHECK:          fir.array_update {{.*}}
! CHECK:        } else {
! CHECK:        }
! CHECK:      }
! CHECK:    }
! CHECK:  } else {
            ! assignment into new storage (`b` unallocated)
! CHECK:    fir.allocmem
! CHECK:    fir.do_loop %{{.*}} {
! CHECK:      fir.array_coor %[[VAL_9]]
! CHECK:      fir.if %{{.*}} {
                ! WHERE
! CHECK:        fir.array_update {{.*}}
! CHECK:      } else {
! CHECK:      }
! CHECK:    }
! CHECK:  }
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.if {{.*}} {
              ! deallocation of `b` old allocatable data store
! CHECK:    }
            ! update of `b` descriptor
! CHECK:  }
          ! Second assignment (in ELSEWHERE)
! CHECK:  fir.do_loop {{.*}} {
! CHECK:    fir.array_coor %[[VAL_9]]{{.*}} : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:    fir.if {{.*}} {
! CHECK:    } else {
              ! elsewhere
! CHECK:      fir.array_update
! CHECK:    }
! CHECK:  }
          ! WHERE temp clean-up
! CHECK:  fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK-NEXT:  return
  where (b > 0)
    b = a
  elsewhere
    b(:) = 0
  end where
end
end module

  use mtest
  integer, allocatable :: a(:), b(:)
  allocate(a(10),b(10))
  a = 5
  b = 1
  call foo(a, b)
  print*, b
  deallocate(a,b)
end
