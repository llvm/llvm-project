! Test copy-in / copy-out of non-contiguous variable passed as F77 array arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Nominal test
! CHECK-LABEL: func @_QPtest_assumed_shape_to_array(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine test_assumed_shape_to_array(x)
  real :: x(:)
! Creating temp
! CHECK:  %[[dim:.*]]:3 = fir.box_dims %[[x:.*]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1 {uniq_name = ".copyinout"}

! Copy-in
! CHECK-DAG:  %[[shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
! CHECK-DAG:  %[[temp_load:.*]] = fir.array_load %[[temp]](%[[shape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK-DAG:  %[[x_load:.*]] = fir.array_load %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:  %[[copyin:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[res:.*]] = %[[temp_load]]) -> (!fir.array<?xf32>) {
! CHECK:    %[[fetch:.*]] = fir.array_fetch %[[x_load]], %[[i]] : (!fir.array<?xf32>, index) -> f32
! CHECK:    %[[update:.*]] = fir.array_update %[[res]], %[[fetch]], %[[i]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:    fir.result %[[update]] : !fir.array<?xf32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[temp_load]], %[[copyin:.*]] to %[[temp]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>

! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPbar(%[[cast]]) : (!fir.ref<!fir.array<?xf32>>) -> ()

! Copy-out

! CHECK-DAG:  %[[x_load:.*]] = fir.array_load %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK-DAG:  %[[shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
! CHECK-DAG:  %[[temp_load:.*]] = fir.array_load %[[temp]](%[[shape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:  %[[copyout:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[res:.*]] = %[[x_load]]) -> (!fir.array<?xf32>) {
! CHECK:    %[[fetch:.*]] = fir.array_fetch %[[temp_load]], %[[i]] : (!fir.array<?xf32>, index) -> f32
! CHECK:    %[[update:.*]] = fir.array_update %[[res]], %[[fetch]], %[[i]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:    fir.result %[[update]] : !fir.array<?xf32>
! CHECK:  }
! CHECK:  fir.array_merge_store %[[x_load]], %[[copyout:.*]] to %[[x]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>

! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>

  call bar(x)
end subroutine

! Test that copy-in/copy-out does not trigger the re-evaluation of
! the designator expression.
! CHECK-LABEL: func @_QPeval_expr_only_once(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<200xf32>>) {
subroutine eval_expr_only_once(x)
  integer :: only_once
  real :: x(200)
! CHECK: fir.call @_QPonly_once()
! CHECK: %[[x_section:.*]] = fir.embox %[[x]](%{{.*}}) [%{{.*}}] : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK-NOT: fir.call @_QPonly_once()
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[temp]]
! CHECK-NOT: fir.call @_QPonly_once()

! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPbar(%[[cast]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar(x(1:200:only_once()))

! CHECK-NOT: fir.call @_QPonly_once()
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[x_section]]
! CHECK-NOT: fir.call @_QPonly_once()
! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>
end subroutine

! Test no copy-in/copy-out is generated for contiguous assumed shapes.
! CHECK-LABEL: func @_QPtest_contiguous(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>
subroutine test_contiguous(x)
  real, contiguous :: x(:)
! CHECK: %[[addr:.*]] = fir.box_addr %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK-NOT:  fir.array_merge_store
! CHECK: fir.call @_QPbar(%[[addr]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar(x)
! CHECK-NOT:  fir.array_merge_store
! CHECK: return
end subroutine

! Test the parenthesis are preventing copy-out.
! CHECK: func @_QPtest_parenthesis(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine test_parenthesis(x)
  real :: x(:)
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1 {uniq_name = ".array.expr"}
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[temp]]
! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPbar(%[[cast]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar((x))
! CHECK-NOT:  fir.array_merge_store
! CHECK: return
end subroutine

! Test copy-in in is skipped for intent(out) arguments.
! CHECK: func @_QPtest_intent_out(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine test_intent_out(x)
  real :: x(:)
  interface
  subroutine bar_intent_out(x)
    real, intent(out) :: x(100)
  end subroutine
  end interface
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1
! CHECK-NOT:  fir.array_merge_store
! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:  fir.call @_QPbar_intent_out(%[[cast]]) : (!fir.ref<!fir.array<100xf32>>) -> ()
  call bar_intent_out(x)
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[x]]
! CHECK: return
end subroutine

! Test copy-out is skipped for intent(out) arguments.
! CHECK: func @_QPtest_intent_in(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine test_intent_in(x)
  real :: x(:)
  interface
  subroutine bar_intent_in(x)
    real, intent(in) :: x(100)
  end subroutine
  end interface
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[temp]]
! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:  fir.call @_QPbar_intent_in(%[[cast]]) : (!fir.ref<!fir.array<100xf32>>) -> ()
  call bar_intent_in(x)
! CHECK-NOT:  fir.array_merge_store
! CHECK: return
end subroutine

! Test copy-in/copy-out is done for intent(inout)
! CHECK: func @_QPtest_intent_inout(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine test_intent_inout(x)
  real :: x(:)
  interface
  subroutine bar_intent_inout(x)
    real, intent(inout) :: x(100)
  end subroutine
  end interface
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[temp]]
! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:  fir.call @_QPbar_intent_inout(%[[cast]]) : (!fir.ref<!fir.array<100xf32>>) -> ()
  call bar_intent_inout(x)
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[x]]
! CHECK: return
end subroutine

! Test characters are handled correctly
! CHECK-LABEL: func @_QPtest_char(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?x!fir.char<1,10>>>) {
subroutine test_char(x)
  character(10) :: x(:)
  ! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
  ! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[dim]]#1 {uniq_name = ".copyinout"}
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[temp_load:.*]] = fir.array_load %[[temp]](%[[shape]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK-DAG: %[[x_load:.*]] = fir.array_load %[[x]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[copy_in:.*]] = fir.do_loop %[[i:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[res:.*]] = %[[temp_load]]) -> (!fir.array<?x!fir.char<1,10>>) {
  ! CHECK:   %[[fetch:.*]] = fir.array_fetch %[[x_load]], %[[i]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK:   %[[update:.*]] = fir.array_update %[[res]], %[[fetch]], %[[i]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>, index) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK:   fir.result %[[update:.*]] : !fir.array<?x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[temp_load]], %[[copy_in]] to %[[temp]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>

  ! CHECK: %[[temp_cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[temp_cast]], %c10{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
  call bar_char(x)

  ! CHECK-DAG: %[[x_load:.*]] = fir.array_load %[[x]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
  ! CHECK-DAG: %[[temp_load:.*]] = fir.array_load %[[temp]](%[[shape]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[copy_out:.*]] = fir.do_loop %[[i:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[res:.*]] = %[[x_load]]) -> (!fir.array<?x!fir.char<1,10>>) {
  ! CHECK:   %[[fetch:.*]] = fir.array_fetch %[[temp_load]], %[[i]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK:   %[[update:.*]] = fir.array_update %[[res]], %[[fetch]], %[[i]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>, index) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK:   fir.result %[[update:.*]] : !fir.array<?x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[x_load]], %[[copy_out]] to %[[x]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_substring_does_no_trigger_copy_inout
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>
subroutine test_scalar_substring_does_no_trigger_copy_inout(c, i, j)
  character(*) :: c
  integer :: i, j
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[c:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[c]], %{{.*}} : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[substr:.*]] = fir.convert %[[coor]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[substr]], %{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char_2(%[[boxchar]]) : (!fir.boxchar<1>) -> ()
  call bar_char_2(c(i:j))
end subroutine
