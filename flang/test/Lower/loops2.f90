! Test loop variables increment
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

module test_loop_var
  implicit none
  integer, pointer:: i_pointer
  integer, allocatable :: i_allocatable
  real, pointer :: x_pointer
  real, allocatable :: x_allocatable
contains
! CHECK-LABEL: func.func @_QMtest_loop_varPtest_pointer
  subroutine test_pointer()
    do i_pointer=1,10
    enddo
! CHECK: %[[PTR:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtest_loop_varEi_pointer"}
! CHECK: %[[BOX:.*]] = fir.load %[[PTR]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[LOOP:.*]] = fir.do_loop
! CHECK:   fir.store %{{.*}} to %[[ADDR]] : !fir.ptr<i32>
! CHECK: fir.store %[[LOOP]] to %[[ADDR]] : !fir.ptr<i32>
  end subroutine

! CHECK-LABEL: func.func @_QMtest_loop_varPtest_allocatable
  subroutine test_allocatable()
    do i_allocatable=1,10
    enddo
! CHECK: %[[ALLOC:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtest_loop_varEi_allocatable"}
! CHECK: %[[BOX:.*]] = fir.load %[[ALLOC]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: %[[LOOP:.*]] = fir.do_loop
! CHECK:   fir.store %{{.*}} to %[[ADDR]] : !fir.heap<i32>
! CHECK: fir.store %[[LOOP]] to %[[ADDR]] : !fir.heap<i32>
  end subroutine

! CHECK-LABEL: func.func @_QMtest_loop_varPtest_real_pointer
  subroutine test_real_pointer()
    do x_pointer=1,10
    enddo
! CHECK: %[[COUNT:.*]] = fir.alloca index
! CHECK: %[[PTR:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtest_loop_varEx_pointer"}
! CHECK: %[[BOX:.*]] = fir.load %[[PTR]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK: fir.store %{{.*}} to %[[ADDR]] : !fir.ptr<f32>
! CHECK: cf.br ^bb1
! CHECK: ^bb1:
! CHECK: cf.cond_br
! CHECK: ^bb2:
! CHECK: %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.ptr<f32>
! CHECK: %[[NEXT:.*]] = arith.addf %[[VAL]], %{{.*}} fastmath<contract> : f32
! CHECK: fir.store %[[NEXT]] to %[[ADDR]] : !fir.ptr<f32>
  end subroutine

! CHECK-LABEL: func.func @_QMtest_loop_varPtest_real_allocatable
  subroutine test_real_allocatable()
    do x_allocatable=1,10
    enddo
! CHECK: %[[COUNT:.*]] = fir.alloca index
! CHECK: %[[ALLOC:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtest_loop_varEx_allocatable"}
! CHECK: %[[BOX:.*]] = fir.load %[[ALLOC]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK: fir.store %{{.*}} to %[[ADDR]] : !fir.heap<f32>
! CHECK: cf.br ^bb1
! CHECK: ^bb1:
! CHECK: cf.cond_br
! CHECK: ^bb2:
! CHECK: %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.heap<f32>
! CHECK: %[[NEXT:.*]] = arith.addf %[[VAL]], %{{.*}} fastmath<contract> : f32
! CHECK: fir.store %[[NEXT]] to %[[ADDR]] : !fir.heap<f32>
  end subroutine

  ! CHECK-LABEL: func.func @_QMtest_loop_varPtest_pointer_unstructured_loop()
  subroutine test_pointer_unstructured_loop()
    do i_pointer=1,10
      if (i_pointer .gt. 5) exit
    enddo
! CHECK: %[[PTR:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtest_loop_varEi_pointer"}
! CHECK: %[[BOX:.*]] = fir.load %[[PTR]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: fir.store %{{.*}} to %[[ADDR]] : !fir.ptr<i32>
! CHECK: cf.br ^bb1
! CHECK: ^bb1:
! CHECK: cf.cond_br
! CHECK: ^bb2:
! CHECK: cf.cond_br
! CHECK: ^bb3:
! CHECK: cf.br ^bb5
! CHECK: ^bb4:
! CHECK: %[[VAL:.*]] = fir.load %[[ADDR]] : !fir.ptr<i32>
! CHECK: %[[NEXT:.*]] = arith.addi %[[VAL]], %{{.*}} overflow<nsw> : i32
! CHECK: fir.store %[[NEXT]] to %[[ADDR]] : !fir.ptr<i32>
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
