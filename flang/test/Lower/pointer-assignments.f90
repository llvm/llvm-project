! Test lowering of pointer assignments
! RUN: bbc --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck %s


! Note that p => NULL() are tested in pointer-disassociate.f90

! -----------------------------------------------------------------------------
!     Test simple pointer assignments to contiguous right-hand side
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}}, %[[x:.*]]: !fir.ref<f32> {{{.*}}, fir.target})
subroutine test_scalar(p, x)
  real, target :: x
  real, pointer :: p
  ! CHECK: %[[box:.*]] = fir.embox %[[x]] : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}}, %[[x:.*]]: !fir.boxchar<1> {{{.*}}, fir.target})
subroutine test_scalar_char(p, x)
  character(*), target :: x
  character(:), pointer :: p
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[box:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[x:.*]]: !fir.ref<!fir.array<100xf32>> {{{.*}}, fir.target})
subroutine test_array(p, x)
  real, target :: x(100)
  real, pointer :: p(:)
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>{{.*}}, %[[x:.*]]: !fir.boxchar<1> {{{.*}}, fir.target}) {
subroutine test_array_char(p, x)
  character(*), target :: x(100)
  character(:), pointer :: p(:)
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[xaddr:.*]] = fir.convert %[[c]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
  ! CHECK-DAG: %[[xaddr2:.*]] = fir.convert %[[xaddr]] : (!fir.ref<!fir.array<100x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %c100{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[xaddr2]](%[[shape]]) typeparams %[[c]]#1
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  p => x
end subroutine

! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from rhs if no bounds spec.
! CHECK-LABEL: func @_QPtest_array_with_lbs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
subroutine test_array_with_lbs(p, x)
  real, target :: x(51:150)
  real, pointer :: p(:)
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c51{{.*}}, %c100{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %{{.*}}(%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! Test that the lhs takes the bounds from rhs.
! CHECK-LABEL: func @_QPtest_pointer_component(
! CHECK-SAME: %[[temp:.*]]: !fir.ref<!fir.type<_QFtest_pointer_componentTmytype{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "temp"}, %[[temp_ptr:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "temp_ptr"}) {
subroutine test_pointer_component(temp, temp_ptr)
  type mytype
    real, pointer :: ptr(:)
  end type mytype
  type(mytype) :: temp
  real, pointer :: temp_ptr(:)
  ! CHECK: %[[ptr_addr:.*]] = fir.coordinate_of %[[temp]], %{{.*}} : (!fir.ref<!fir.type<_QFtest_pointer_componentTmytype{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[ptr:.*]] = fir.load %[[ptr_addr]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[dims:.*]]:3 = fir.box_dims %[[ptr]], %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[shift:.*]] = fir.shift %[[dims]]#0 : (index) -> !fir.shift<1>
  ! CHECK: %[[arr_box:.*]] = fir.rebox %[[ptr]](%[[shift]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[shift2:.*]] = fir.shift %[[dims]]#0 : (index) -> !fir.shift<1>
  ! CHECK: %[[final_box:.*]] = fir.rebox %[[arr_box]](%[[shift2]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[final_box]] to %[[temp_ptr]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  temp_ptr => temp%ptr
end subroutine

! -----------------------------------------------------------------------------
!    Test pointer assignments with bound specs to contiguous right-hand side
! -----------------------------------------------------------------------------

! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from bound spec if specified
! CHECK-LABEL: func @_QPtest_array_with_new_lbs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
subroutine test_array_with_new_lbs(p, x)
  real, target :: x(51:150)
  real, pointer :: p(:)
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c4{{.*}}, %c100{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %{{.*}}(%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p(4:) => x
end subroutine

! Test F2018 10.2.2.3 point 9: bounds remapping
! CHECK-LABEL: func @_QPtest_array_remap(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>{{.*}}, %[[x:.*]]: !fir.ref<!fir.array<100xf32>> {{{.*}}, fir.target})
subroutine test_array_remap(p, x)
  real, target :: x(100)
  real, pointer :: p(:, :)
  ! CHECK-DAG: %[[c2_idx:.*]] = fir.convert %c2{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[c11_idx:.*]] = fir.convert %c11{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[diff0:.*]] = arith.subi %[[c11_idx]], %[[c2_idx]] : index
  ! CHECK-DAG: %[[ext0:.*]] = arith.addi %[[diff0:.*]], %c1{{.*}} : index
  ! CHECK-DAG: %[[c3_idx:.*]] = fir.convert %c3{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[c12_idx:.*]] = fir.convert %c12{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[diff1:.*]] = arith.subi %[[c12_idx]], %[[c3_idx]] : index
  ! CHECK-DAG: %[[ext1:.*]] = arith.addi %[[diff1]], %c1{{.*}} : index
  ! CHECK-DAG: %[[addrCast:.*]] = fir.convert %[[x]] : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c2{{.*}}, %[[ext0]], %c3{{.*}}, %[[ext1]]
  ! CHECK: %[[box:.*]] = fir.embox %[[addrCast]](%[[shape]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_char_remap(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>{{.*}}, %[[x:.*]]: !fir.boxchar<1> {{{.*}}, fir.target})
subroutine test_array_char_remap(p, x)
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[x]]
  character(*), target :: x(100)
  character(:), pointer :: p(:, :)
  ! CHECK: subi
  ! CHECK: %[[ext0:.*]] = arith.addi
  ! CHECK: subi
  ! CHECK: %[[ext1:.*]] = arith.addi
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c2{{.*}}, %[[ext0]], %c3{{.*}}, %[[ext1]]
  ! CHECK: %[[box:.*]] = fir.embox %{{.*}}(%[[shape]]) typeparams %[[unbox]]#1 : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shapeshift<2>, index) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK: fir.store %[[box]] to %[[p]]
  p(2:11, 3:12) => x
end subroutine

! -----------------------------------------------------------------------------
!  Test simple pointer assignments to non contiguous right-hand side
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_array_non_contig_rhs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[x:.*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_rhs(p, x)
  real, target :: x(:)
  real, pointer :: p(:)
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from rhs if no bounds spec.
! CHECK-LABEL: func @_QPtest_array_non_contig_rhs_lbs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[x:.*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_rhs_lbs(p, x)
  real, target :: x(7:)
  real, pointer :: p(:)
  ! CHECK: %[[c7_idx:.*]] = fir.convert %c7{{.*}} : (i64) -> index
  ! CHECK: %[[shift:.*]] = fir.shift %[[c7_idx]] : (index) -> !fir.shift<1>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]](%[[shift]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_non_contig_rhs2(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<200xf32>> {{{.*}}, fir.target}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 200 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 160 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.slice %[[VAL_4]], %[[VAL_8]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_1]](%[[VAL_9]]) {{\[}}%[[VAL_10]]] : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<51xf32>>
! CHECK:         %[[VAL_12:.*]] = fir.rebox %[[VAL_11]] : (!fir.box<!fir.array<51xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:         return
! CHECK:       }

subroutine test_array_non_contig_rhs2(p, x)
  real, target :: x(200)
  real, pointer :: p(:)
  p => x(10:160:3)
end subroutine

! -----------------------------------------------------------------------------
!  Test pointer assignments with bound specs to non contiguous right-hand side
! -----------------------------------------------------------------------------


! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from bound spec if specified
! CHECK-LABEL: func @_QPtest_array_non_contig_rhs_new_lbs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[x:.*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_rhs_new_lbs(p, x)
  real, target :: x(7:)
  real, pointer :: p(:)
  ! CHECK: %[[shift:.*]] = fir.shift %c4{{.*}}
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]](%[[shift]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>

  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p(4:) => x
end subroutine

! Test F2018 10.2.2.3 point 9: bounds remapping
! CHECK-LABEL: func @_QPtest_array_non_contig_remap(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>{{.*}}, %[[x:.*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_remap(p, x)
  real, target :: x(:)
  real, pointer :: p(:, :)
  ! CHECK: subi
  ! CHECK: %[[ext0:.*]] = arith.addi
  ! CHECK: subi
  ! CHECK: %[[ext1:.*]] = arith.addi
  ! CHECK: %[[shape:.*]] = fir.shape_shift %{{.*}}, %[[ext0]], %{{.*}}, %[[ext1]]
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]](%[[shape]]) : (!fir.box<!fir.array<?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x
end subroutine

! Test remapping a slice

! CHECK-LABEL: func @_QPtest_array_non_contig_remap_slice(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<400xf32>> {{{.*}}, fir.target}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 400 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_4:.*]] = arith.constant 11 : i64
! CHECK:         %[[VAL_5:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_6:.*]] = arith.constant 12 : i64
! CHECK:         %[[VAL_7:.*]] = arith.constant 51 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 350 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.slice %[[VAL_8]], %[[VAL_12]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_15:.*]] = fir.embox %[[VAL_1]](%[[VAL_13]]) {{\[}}%[[VAL_14]]] : (!fir.ref<!fir.array<400xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<100xf32>>
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_17]] : index
! CHECK:         %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_16]] : index
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_21]] : index
! CHECK:         %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_16]] : index
! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_27:.*]] = fir.shape_shift %[[VAL_25]], %[[VAL_20]], %[[VAL_26]], %[[VAL_24]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_28:.*]] = fir.rebox %[[VAL_15]](%[[VAL_27]]) : (!fir.box<!fir.array<100xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[VAL_28]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:         return
! CHECK:       }
subroutine test_array_non_contig_remap_slice(p, x)
  real, target :: x(400)
  real, pointer :: p(:, :)
  p(2:11, 3:12) => x(51:350:3)
end subroutine

! -----------------------------------------------------------------------------
!  Test pointer assignments that involves LHS pointers lowered to local variables
!  instead of a fir.ref<fir.box>, and RHS that are fir.box
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPissue857(
! CHECK-SAME: %[[rhs:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>>
subroutine issue857(rhs)
  type t
    integer :: i
  end type
  type(t), pointer :: rhs, lhs
  ! CHECK: %[[lhs:.*]] = fir.alloca !fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>
  ! CHECK: %[[box_load:.*]] = fir.load %[[rhs]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box_load]] : (!fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>) -> !fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>
  ! CHECK: fir.store %[[addr]] to %[[lhs]] : !fir.ref<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>
  lhs => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_array(
! CHECK-SAME: %[[rhs:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>>
subroutine issue857_array(rhs)
  type t
    integer :: i
  end type
  type(t), contiguous,  pointer :: rhs(:), lhs(:)
  ! CHECK-DAG: %[[lhs_addr:.*]] = fir.alloca !fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>> {uniq_name = "_QFissue857_arrayElhs.addr"}
  ! CHECK-DAG: %[[lhs_lb:.*]] = fir.alloca index {uniq_name = "_QFissue857_arrayElhs.lb0"}
  ! CHECK-DAG: %[[lhs_ext:.*]] = fir.alloca index {uniq_name = "_QFissue857_arrayElhs.ext0"}
  ! CHECK: %[[box:.*]] = fir.load %[[rhs]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>>
  ! CHECK: %[[lb:.*]]:3 = fir.box_dims %[[box]], %c{{.*}} : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>, index) -> (index, index, index)
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>) -> !fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>
  ! CHECK: %[[ext:.*]]:3 = fir.box_dims %[[box]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>, index) -> (index, index, index)
  ! CHECK-DAG: fir.store %[[addr]] to %[[lhs_addr]] : !fir.ref<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>
  ! CHECK-DAG: fir.store %[[ext]]#1 to %[[lhs_ext]] : !fir.ref<index>
  ! CHECK-DAG: fir.store %[[lb]]#0 to %[[lhs_lb]] : !fir.ref<index>
  lhs => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_array_shift(
subroutine issue857_array_shift(rhs)
  ! Test lower bounds is the one from the shift
  type t
    integer :: i
  end type
  type(t), contiguous,  pointer :: rhs(:), lhs(:)
  ! CHECK: %[[lhs_lb:.*]] = fir.alloca index {uniq_name = "_QFissue857_array_shiftElhs.lb0"}
  ! CHECK: %[[c42:.*]] = fir.convert %c42{{.*}} : (i64) -> index
  ! CHECK: fir.store %[[c42]] to %[[lhs_lb]] : !fir.ref<index>
  lhs(42:) => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_array_remap
subroutine issue857_array_remap(rhs)
  ! Test lower bounds is the one from the shift
  type t
    integer :: i
  end type
  type(t), contiguous,  pointer :: rhs(:, :), lhs(:)
  ! CHECK-DAG: %[[lhs_addr:.*]] = fir.alloca !fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_remapTt{i:i32}>>> {uniq_name = "_QFissue857_array_remapElhs.addr"}
  ! CHECK-DAG: %[[lhs_lb:.*]] = fir.alloca index {uniq_name = "_QFissue857_array_remapElhs.lb0"}
  ! CHECK-DAG: %[[lhs_ext:.*]] = fir.alloca index {uniq_name = "_QFissue857_array_remapElhs.ext0"}

  ! CHECK: %[[c101:.*]] = fir.convert %c101_i64 : (i64) -> index
  ! CHECK: %[[c200:.*]] = fir.convert %c200_i64 : (i64) -> index
  ! CHECK: %[[sub:.*]] = arith.subi %[[c200]], %[[c101]] : index
  ! CHECK: %[[extent:.*]] = arith.addi %[[sub]], %c1{{.*}} : index
  ! CHECK: %[[addr:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>>) -> !fir.ptr<!fir.array<?x?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>
  ! CHECK: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.array<?x?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>) -> !fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>
  ! CHECK: fir.store %[[addr_cast]] to %[[lhs_addr]] : !fir.ref<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>>
  ! CHECK: fir.store %[[extent]] to %[[lhs_ext]] : !fir.ref<index>
  ! CHECK: %[[c101_2:.*]] = fir.convert %c101{{.*}} : (i64) -> index
  ! CHECK: fir.store %[[c101_2]] to %[[lhs_lb]] : !fir.ref<index>
  lhs(101:200) => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_char
subroutine issue857_char(rhs)
  ! Only check that the length is taken from the fir.box created for the slice.
  ! CHECK-DAG: %[[lhs1_len:.*]] = fir.alloca index {uniq_name = "_QFissue857_charElhs1.len"}
  ! CHECK-DAG: %[[lhs2_len:.*]] = fir.alloca index {uniq_name = "_QFissue857_charElhs2.len"}
  character(:), contiguous,  pointer ::  lhs1(:), lhs2(:, :)
  character(*), target ::  rhs(100)
  ! CHECK: %[[len:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.array<50x!fir.char<1,?>>>) -> index
  ! CHECK: fir.store %[[len]] to %[[lhs1_len]] : !fir.ref<index>
  lhs1 => rhs(1:50:1)
  ! CHECK: %[[len2:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.array<50x!fir.char<1,?>>>) -> index
  ! CHECK: fir.store %[[len2]] to %[[lhs2_len]] : !fir.ref<index>
  lhs2(1:2, 1:25) => rhs(1:50:1)
end subroutine

! CHECK-LABEL: func @_QPissue1180(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {{{.*}}, fir.target}) {
subroutine issue1180(x)
  integer, target :: x
  integer, pointer :: p
  common /some_common/ p
  ! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QCsome_common) : !fir.ref<!fir.array<24xi8>>
  ! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<24xi8>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK: %[[VAL_3:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK: %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  ! CHECK: fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
  p => x
end subroutine
