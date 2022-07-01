! Test copy-in / copy-out of non-contiguous variable passed as F77 array arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Nominal test
! CHECK-LABEL: func @_QPtest_assumed_shape_to_array(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
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
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<200xf32>>{{.*}}) {
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
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_parenthesis(x)
  real :: x(:)
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1 {uniq_name = ".array.expr"}
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[temp]]
! CHECK:  %[[cast:.*]] = fir.convert %[[temp]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPbar(%[[cast]]) : (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar((x))
! CHECK-NOT:  fir.array_merge_store
! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>
! CHECK: return
end subroutine

! Test copy-in in is skipped for intent(out) arguments.
! CHECK: func @_QPtest_intent_out(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
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
! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>
! CHECK: return
end subroutine

! Test copy-out is skipped for intent(out) arguments.
! CHECK: func @_QPtest_intent_in(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
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
! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>
! CHECK: return
end subroutine

! Test copy-in/copy-out is done for intent(inout)
! CHECK: func @_QPtest_intent_inout(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
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
! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<?xf32>>
! CHECK: return
end subroutine

! Test characters are handled correctly
! CHECK-LABEL: func @_QPtest_char(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,10>>>{{.*}}) {
subroutine test_char(x)
  ! CHECK: %[[VAL_1:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_2:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
  ! CHECK: %[[VAL_4:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_3]]#1 {uniq_name = ".copyinout"}
  ! CHECK: %[[VAL_5:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[VAL_7:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[VAL_8:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_9:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_10:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_8]] : index
  ! CHECK: %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_6]]) -> (!fir.array<?x!fir.char<1,10>>) {
  ! CHECK: %[[VAL_14:.*]] = fir.array_access %[[VAL_7]], %[[VAL_12]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_15:.*]] = fir.array_access %[[VAL_13]], %[[VAL_12]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_16:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_17:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i64
  ! CHECK: %[[VAL_19:.*]] = arith.muli %[[VAL_17]], %[[VAL_18]] : i64
  ! CHECK: %[[VAL_20:.*]] = arith.constant false
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_21]], %[[VAL_22]], %[[VAL_19]], %[[VAL_20]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_23:.*]] = fir.array_amend %[[VAL_13]], %[[VAL_15]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: fir.result %[[VAL_23]] : !fir.array<?x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_6]], %[[VAL_24:.*]] to %[[VAL_4]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_26:.*]] = fir.emboxchar %[[VAL_25]], %[[VAL_1]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char(%[[VAL_26]]) : (!fir.boxchar<1>) -> ()
  ! CHECK: %[[VAL_27:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[VAL_28:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_28]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
  ! CHECK: %[[VAL_30:.*]] = fir.shape %[[VAL_3]]#1 : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_31:.*]] = fir.array_load %[[VAL_4]](%[[VAL_30]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: %[[VAL_32:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_33:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_34:.*]] = arith.subi %[[VAL_29]]#1, %[[VAL_32]] : index
  ! CHECK: %[[VAL_35:.*]] = fir.do_loop %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_34]] step %[[VAL_32]] unordered iter_args(%[[VAL_37:.*]] = %[[VAL_27]]) -> (!fir.array<?x!fir.char<1,10>>) {
  ! CHECK: %[[VAL_38:.*]] = fir.array_access %[[VAL_31]], %[[VAL_36]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_39:.*]] = fir.array_access %[[VAL_37]], %[[VAL_36]] : (!fir.array<?x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_40:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_41:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_40]] : (index) -> i64
  ! CHECK: %[[VAL_43:.*]] = arith.muli %[[VAL_41]], %[[VAL_42]] : i64
  ! CHECK: %[[VAL_44:.*]] = arith.constant false
  ! CHECK: %[[VAL_45:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_46:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_45]], %[[VAL_46]], %[[VAL_43]], %[[VAL_44]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_47:.*]] = fir.array_amend %[[VAL_37]], %[[VAL_39]] : (!fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<?x!fir.char<1,10>>
  ! CHECK: fir.result %[[VAL_47]] : !fir.array<?x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_27]], %[[VAL_48:.*]] to %[[VAL_0]] : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
  ! CHECK: fir.freemem %[[VAL_4]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>

  character(10) :: x(:)
  call bar_char(x)
  ! CHECK:         return
  ! CHECK:       }
end subroutine test_char

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

! CHECK-LABEL: func @_QPissue871(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFissue871Tt{i:i32}>>>>{{.*}})
subroutine issue871(p)
  ! Test passing implicit derived from scalar pointer (no copy-in/out).
  type t
    integer :: i
  end type t
  type(t), pointer :: p
  ! CHECK: %[[box_load:.*]] = fir.load %[[p]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box_load]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: fir.call @_QPbar_derived(%[[cast]])
  call bar_derived(p)
end subroutine

! CHECK-LABEL: func @_QPissue871_array(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue871_arrayTt{i:i32}>>>>>
subroutine issue871_array(p)
  ! Test passing implicit derived from contiguous pointer (no copy-in/out).
  type t
    integer :: i
  end type t
  type(t), pointer, contiguous :: p(:)
  ! CHECK: %[[box_load:.*]] = fir.load %[[p]]
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box_load]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]]
  ! CHECK: fir.call @_QPbar_derived_array(%[[cast]])
  call bar_derived_array(p)
end subroutine

! CHECK-LABEL: func @_QPwhole_components()
subroutine whole_components()
  ! Test no copy is made for whole components.
  type t
    integer :: i(100)
  end type
  ! CHECK: %[[a:.*]] = fir.alloca !fir.type<_QFwhole_componentsTt{i:!fir.array<100xi32>}>
  type(t) :: a
  ! CHECK: %[[field:.*]] = fir.field_index i, !fir.type<_QFwhole_componentsTt{i:!fir.array<100xi32>}>
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[a]], %[[field]] : (!fir.ref<!fir.type<_QFwhole_componentsTt{i:!fir.array<100xi32>}>>, !fir.field) -> !fir.ref<!fir.array<100xi32>>
  ! CHECK: fir.call @_QPbar_integer(%[[addr]]) : (!fir.ref<!fir.array<100xi32>>) -> ()
  call bar_integer(a%i)
end subroutine

! CHECK-LABEL: func @_QPwhole_component_contiguous_pointer()
subroutine whole_component_contiguous_pointer()
  ! Test no copy is made for whole contiguous pointer components.
  type t
    integer, pointer, contiguous :: i(:)
  end type
  ! CHECK: %[[a:.*]] = fir.alloca !fir.type<_QFwhole_component_contiguous_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
  type(t) :: a
  ! CHECK: %[[field:.*]] = fir.field_index i, !fir.type<_QFwhole_component_contiguous_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a]], %[[field]] : (!fir.ref<!fir.type<_QFwhole_component_contiguous_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK: %[[box_load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.array<?xi32>>) -> !fir.ref<!fir.array<100xi32>>
  ! CHECK: fir.call @_QPbar_integer(%[[cast]]) : (!fir.ref<!fir.array<100xi32>>) -> ()
  call bar_integer(a%i)
end subroutine

! CHECK-LABEL: func @_QPwhole_component_contiguous_char_pointer()
subroutine whole_component_contiguous_char_pointer()
  ! Test no copy is made for whole contiguous character pointer components.
  type t
    character(:), pointer, contiguous :: i(:)
  end type
  ! CHECK: %[[a:.*]] = fir.alloca !fir.type<_QFwhole_component_contiguous_char_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>
  type(t) :: a
  ! CHECK: %[[field:.*]] = fir.field_index i, !fir.type<_QFwhole_component_contiguous_char_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QFwhole_component_contiguous_char_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[box_load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[box_load]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[embox:.*]] = fir.emboxchar %[[cast]], %[[len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char_3(%[[embox]]) : (!fir.boxchar<1>) -> ()
  call bar_char_3(a%i)
end subroutine
