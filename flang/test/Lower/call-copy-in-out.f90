! Test copy-in / copy-out of non-contiguous variable passed as F77 array arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Nominal test
! CHECK-LABEL: func @_QPtest_assumed_shape_to_array(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_assumed_shape_to_array(x)
  real :: x(:)

! CHECK: %[[box_none:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK: %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK: %[[addr:.*]] = fir.if %[[is_contiguous]] -> (!fir.heap<!fir.array<?xf32>>) {
! CHECK:   %[[box_addr:.*]] = fir.box_addr %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:   fir.result %[[box_addr]] : !fir.heap<!fir.array<?xf32>>
! CHECK: } else {
! Creating temp
! CHECK:  %[[dim:.*]]:3 = fir.box_dims %[[x:.*]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1 {uniq_name = ".copyinout"}

! Copy-in
! CHECK-DAG:  %[[shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
! CHECK-DAG:  %[[temp_box:.*]] = fir.embox %[[temp]](%[[shape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK-DAG:  fir.store %[[temp_box]] to %[[temp_box_loc:.*]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK-DAG: %[[temp_box_addr:.*]] = fir.convert %[[temp_box_loc]] : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG: %[[arg_box:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK-DAG: fir.call @_FortranAAssignTemporary(%[[temp_box_addr]], %[[arg_box]], %{{.*}}, %{{.*}}){{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:  fir.result %[[temp]] : !fir.heap<!fir.array<?xf32>>

! CHECK:  %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPbar(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()

! Copy-out
! CHECK-DAG:  %[[shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
! CHECK-DAG:  %[[temp_box:.*]] = fir.embox %[[addr]](%[[shape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK-DAG:  fir.store %[[x]] to %[[arg_box_loc:.*]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK-DAG:  %[[skipToInit:.*]] = arith.constant true
! CHECK-DAG: %[[arg_box_addr:.*]] = fir.convert %[[arg_box_loc]] : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG: %[[temp_box_cast:.*]] = fir.convert %[[temp_box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK-DAG: fir.call @_FortranACopyOutAssign(%[[arg_box_addr]], %[[temp_box_cast]], %[[skipToInit]], %{{.*}}, %{{.*}}){{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.ref<i8>, i32) -> none
! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>

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
! CHECK: %[[box_none:.*]] = fir.convert %[[x_section]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK: %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK: %[[addr:.*]] = fir.if %[[is_contiguous]] -> (!fir.heap<!fir.array<?xf32>>) {

! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>
! CHECK-NOT: fir.call @_QPonly_once()
! CHECK:  fir.call @_FortranAAssignTemporary
! CHECK-NOT: fir.call @_QPonly_once()

! CHECK:  %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPbar(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar(x(1:200:only_once()))

! CHECK-NOT: fir.call @_QPonly_once()
! CHECK:  fir.call @_FortranACopyOutAssign
! CHECK-NOT: fir.call @_QPonly_once()

! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>
end subroutine

! Test no copy-in/copy-out is generated for contiguous assumed shapes.
! CHECK-LABEL: func @_QPtest_contiguous(
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>
subroutine test_contiguous(x)
  real, contiguous :: x(:)
! CHECK: %[[addr:.*]] = fir.box_addr %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK-NOT:  fir.call @_FortranAAssignTemporary
! CHECK: fir.call @_QPbar(%[[addr]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar(x)
! CHECK-NOT:  fir.call @_FortranACopyOutAssign
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
! CHECK:  fir.call @_QPbar(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
  call bar((x))
! CHECK-NOT:  fir.call @_FortranACopyOutAssign
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
! CHECK: %[[box_none:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK: %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK: %[[addr:.*]] = fir.if %[[is_contiguous]]
! CHECK: } else {
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1
! CHECK-NOT:  fir.call @_FortranAAssignTemporary
! CHECK: %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false{{.*}} : i1
! CHECK:  %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:  fir.call @_QPbar_intent_out(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call bar_intent_out(x)
  
! CHECK: fir.if %[[not_contiguous]]
! CHECK: fir.call @_FortranACopyOutAssign
! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>
! CHECK: return
end subroutine

! Test copy-out is skipped for intent(out) arguments.
! CHECK-LABEL: func.func @_QPtest_intent_in(
! CHECK: %[[x:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_intent_in(x)
  real :: x(:)
  interface
  subroutine bar_intent_in(x)
    real, intent(in) :: x(100)
  end subroutine
  end interface
! CHECK: %[[box_none:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK: %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK: %[[addr:.*]] = fir.if %[[is_contiguous]]
! CHECK: } else {
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1
! CHECK: %[[temp_shape:.*]] = fir.shape %[[dim]]#1 : (index) -> !fir.shape<1>
! CHECK: %[[temp_box:.*]] = fir.embox %[[temp]](%[[temp_shape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK: fir.store %[[temp_box]] to %[[temp_box_loc:.*]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK: %[[temp_box_addr:.*]] = fir.convert %[[temp_box_loc]] : (!fir.ref<!fir.box<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @_FortranAAssignTemporary(%[[temp_box_addr]],
! CHECK: %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false{{.*}} : i1
! CHECK:  %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:  fir.call @_QPbar_intent_in(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call bar_intent_in(x)
! CHECK: fir.if %[[not_contiguous]]
! CHECK-NOT:  fir.call @_FortranACopyOutAssign
! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>
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
! CHECK: %[[box_none:.*]] = fir.convert %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK: %[[is_contiguous:.*]] = fir.call @_FortranAIsContiguous(%[[box_none]]) {{.*}}: (!fir.box<none>) -> i1
! CHECK: %[[addr:.*]] = fir.if %[[is_contiguous]]
! CHECK: } else {
! CHECK: %[[dim:.*]]:3 = fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xf32>, %[[dim]]#1
! CHECK:  fir.call @_FortranAAssign
! CHECK: %[[not_contiguous:.*]] = arith.cmpi eq, %[[is_contiguous]], %false{{.*}} : i1
! CHECK:  %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:  fir.call @_QPbar_intent_inout(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<100xf32>>) -> ()
  call bar_intent_inout(x)
! CHECK: fir.if %[[not_contiguous]]
! CHECK:  fir.call @_FortranACopyOutAssign
! CHECK: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>
! CHECK: return
end subroutine

! Test characters are handled correctly
! CHECK-LABEL: func @_QPtest_char(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,10>>>{{.*}}) {
subroutine test_char(x)
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.array<?x!fir.char<1,10>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.array<?x!fir.char<1,10>>>
! CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<none>
! CHECK:           %[[VAL_5:.*]] = fir.call @_FortranAIsContiguous(%[[VAL_4]]) fastmath<contract> : (!fir.box<none>) -> i1
! CHECK:           %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (!fir.heap<!fir.array<?x!fir.char<1,10>>>) {
! CHECK:             %[[VAL_7:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:             fir.result %[[VAL_7]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           } else {
! CHECK:             %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_8]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
! CHECK:             %[[VAL_10:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_9]]#1 {uniq_name = ".copyinout"}
! CHECK:             %[[VAL_11:.*]] = fir.shape %[[VAL_9]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_12:.*]] = fir.embox %[[VAL_10]](%[[VAL_11]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
! CHECK:             fir.store %[[VAL_12]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?x!fir.char<1,10>>>>
! CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.array<?x!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<none>
! CHECK:             %[[VAL_18:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_15]], %[[VAL_16]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:             fir.result %[[VAL_10]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           }
! CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_19]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_21:.*]] = arith.constant false
! CHECK:           %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_21]] : i1
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_24:.*]] : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_25:.*]] = fir.emboxchar %[[VAL_23]], %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPbar_char(%[[VAL_25]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           fir.if %[[VAL_22]] {
! CHECK:             %[[VAL_26:.*]] = fir.shape %[[VAL_20]]#1 : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_27:.*]] = fir.embox %[[VAL_24]](%[[VAL_26]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
! CHECK:             fir.store %[[VAL_0]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.array<?x!fir.char<1,10>>>>
! CHECK:             %[[VAL_30:.*]] = arith.constant true
! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.array<?x!fir.char<1,10>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_27]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> !fir.box<none>
! CHECK:             %[[VAL_34:.*]] = fir.call @_FortranACopyOutAssign(%[[VAL_31]], %[[VAL_32]], %[[VAL_30]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.ref<i8>, i32) -> none
! CHECK:             fir.freemem %[[VAL_24]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:           }
  
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
  ! CHECK: fir.call @_QPbar_char_2(%[[boxchar]]) {{.*}}: (!fir.boxchar<1>) -> ()
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
  ! CHECK: fir.call @_QPbar_integer(%[[addr]]) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
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
  ! CHECK: fir.call @_QPbar_integer(%[[cast]]) {{.*}}: (!fir.ref<!fir.array<100xi32>>) -> ()
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
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[a]], %[[field]] : (!fir.ref<!fir.type<_QFwhole_component_contiguous_char_pointerTt{i:!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[box_load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[box_load]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>) -> index
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[embox:.*]] = fir.emboxchar %[[cast]], %[[len]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar_char_3(%[[embox]]) {{.*}}: (!fir.boxchar<1>) -> ()
  call bar_char_3(a%i)
end subroutine
