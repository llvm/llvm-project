! Test loop variables increment
! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

module test_loop_var
  implicit none
  integer, pointer:: i_pointer
  integer, allocatable :: i_allocatable
  real, pointer :: x_pointer
  real, allocatable :: x_allocatable
contains
! CHECK-LABEL: func @_QMtest_loop_varPtest_pointer
  subroutine test_pointer()
    do i_pointer=1,10
    enddo
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QMtest_loop_varEi_pointer) : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_9:.*]]:2 = fir.do_loop{{.*}}iter_args(%[[IV:.*]] = {{.*}})
! CHECK:           fir.store %[[IV]] to %[[VAL_2]] : !fir.ptr<i32>
! CHECK:         }
! CHECK:         fir.store %[[VAL_9]]#1 to %[[VAL_2]] : !fir.ptr<i32>
  end subroutine

! CHECK-LABEL: func @_QMtest_loop_varPtest_allocatable
  subroutine test_allocatable()
    do i_allocatable=1,10
    enddo
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QMtest_loop_varEi_allocatable) : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:         %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:         %[[VAL_9:.*]]:2 = fir.do_loop{{.*}}iter_args(%[[IV:.*]] = {{.*}})
! CHECK:           fir.store %[[IV]] to %[[VAL_2]] : !fir.heap<i32>
! CHECK:         }
! CHECK:         fir.store %[[VAL_9]]#1 to %[[VAL_2]] : !fir.heap<i32>
  end subroutine

! CHECK-LABEL: func @_QMtest_loop_varPtest_real_pointer
  subroutine test_real_pointer()
    do x_pointer=1,10
    enddo
! CHECK:         %[[VAL_0:.*]] = fir.alloca index
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QMtest_loop_varEx_pointer) : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> f32
! CHECK:         %[[VAL_8:.*]] = arith.constant 1.000000e+00 : f32

! CHECK:         fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ptr<f32>
! CHECK:         br ^bb1
! CHECK:       ^bb1:
! CHECK:         cond_br %{{.*}}, ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.ptr<f32>
! CHECK:         %[[VAL_20:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:         %[[VAL_21:.*]] = arith.addf %[[VAL_19]], %[[VAL_20]] {{.*}}: f32
! CHECK:         fir.store %[[VAL_21]] to %[[VAL_3]] : !fir.ptr<f32>
! CHECK:         br ^bb1
! CHECK:       ^bb3:
! CHECK:         return
  end subroutine

! CHECK-LABEL: func @_QMtest_loop_varPtest_real_allocatable
  subroutine test_real_allocatable()
    do x_allocatable=1,10
    enddo
! CHECK:         %[[VAL_0:.*]] = fir.alloca index
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QMtest_loop_varEx_allocatable) : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> f32
! CHECK:         %[[VAL_8:.*]] = arith.constant 1.000000e+00 : f32

! CHECK:         fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.heap<f32>
! CHECK:         br ^bb1
! CHECK:       ^bb1:
! CHECK:         cond_br %{{.*}}, ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_19:.*]] = fir.load %[[VAL_3]] : !fir.heap<f32>
! CHECK:         %[[VAL_20:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:         %[[VAL_21:.*]] = arith.addf %[[VAL_19]], %[[VAL_20]] {{.*}}: f32
! CHECK:         fir.store %[[VAL_21]] to %[[VAL_3]] : !fir.heap<f32>
! CHECK:         br ^bb1
! CHECK:       ^bb3:
! CHECK:         return
  end subroutine

  ! CHECK-LABEL: func @_QMtest_loop_varPtest_pointer_unstructured_loop()
  subroutine test_pointer_unstructured_loop()
    do i_pointer=1,10
      if (i_pointer .gt. 5) exit
    enddo
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QMtest_loop_varEi_pointer) : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_2:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         fir.store %[[VAL_4]] to %[[VAL_3]] : !fir.ptr<i32>
! CHECK:         br ^bb1
! CHECK:       ^bb1:
! CHECK:         cond_br %{{.*}}, ^bb2, ^bb5
! CHECK:       ^bb2:
! CHECK:         cond_br %{{.*}}, ^bb3, ^bb4
! CHECK:       ^bb3:
! CHECK:         br ^bb5
! CHECK:       ^bb4:
! CHECK:         %[[VAL_20:.*]] = fir.load %[[VAL_3]] : !fir.ptr<i32>
! CHECK:         %[[VAL_21:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] overflow<nsw> : i32
! CHECK:         fir.store %[[VAL_22]] to %[[VAL_3]] : !fir.ptr<i32>
! CHECK:         br ^bb1
! CHECK:       ^bb5:
! CHECK:         return
! CHECK:       }
  end subroutine

end module

  use test_loop_var
  implicit none
  integer, target :: i_target = -1
  real, target :: x_target = -1.
  i_pointer => i_target
  allocate(i_allocatable)
  i_allocatable = -1
  x_pointer => x_target
  allocate(x_allocatable)
  x_allocatable = -1.

  call test_pointer()
  call test_allocatable()
  call test_real_pointer()
  call test_real_allocatable()
  ! Expect 11 everywhere
  print *, i_target
  print *, i_allocatable
  print *, x_target
  print *, x_allocatable

  call test_pointer_unstructured_loop()
  call test_allocatable_unstructured_loop()
  ! Expect 6 everywhere
  print *, i_target
end
