! Test lowering of pointer assignments
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s


! Note that p => NULL() are tested in pointer-disassociate.f90

! -----------------------------------------------------------------------------
!     Test simple pointer assignments to contiguous right-hand side
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}}, %[[arg1:[^:]*]]: !fir.ref<f32> {{{.*}}, fir.target})
subroutine test_scalar(p, x)
  real, target :: x
  real, pointer :: p
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.boxchar<1> {{{.*}}, fir.target})
subroutine test_scalar_char(p, x)
  character(*), target :: x
  character(:), pointer :: p
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %[[arg1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[c]]#0 typeparams %[[c]]#1
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[x]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[box:.*]] = fir.embox %[[unbox]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.ref<!fir.array<100xf32>> {{{.*}}, fir.target})
subroutine test_array(p, x)
  real, target :: x(100)
  real, pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}}
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]](%[[shape]])
  ! CHECK: %[[addr:.*]] = fir.convert %[[x]]#0 : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[box:.*]] = fir.embox %[[addr]](%[[shape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_char(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.boxchar<1> {{{.*}}, fir.target}) {
subroutine test_array_char(p, x)
  character(*), target :: x(100)
  character(:), pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %[[arg1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) typeparams %[[c]]#1
  ! CHECK: %[[box:.*]] = fir.rebox %[[x]]#0 : (!fir.box<!fir.array<100x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>
  p => x
end subroutine

! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from rhs if no bounds spec.
! CHECK-LABEL: func @_QPtest_array_with_lbs(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
subroutine test_array_with_lbs(p, x)
  real, target :: x(51:150)
  real, pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[ss:.*]] = fir.shape_shift %c51{{.*}}, %c100{{.*}}
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %{{.*}}(%[[ss]])
  ! CHECK: %[[shift:.*]] = fir.shift %c51{{.*}} : (index) -> !fir.shift<1>
  ! CHECK: %[[box:.*]] = fir.rebox %[[x]]#0(%[[shift]]) : (!fir.box<!fir.array<100xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! Test that the lhs takes the bounds from rhs.
! CHECK-LABEL: func @_QPtest_pointer_component(
! CHECK-SAME: %[[arg_temp:[^:]*]]: !fir.ref<!fir.type<_QFtest_pointer_componentTmytype{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "temp"}, %[[arg_temp_ptr:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "temp_ptr"}) {
subroutine test_pointer_component(temp, temp_ptr)
  type mytype
    real, pointer :: ptr(:)
  end type mytype
  type(mytype) :: temp
  real, pointer :: temp_ptr(:)
  ! CHECK: %[[temp:.*]]:2 = hlfir.declare %[[arg_temp]]
  ! CHECK: %[[temp_ptr:.*]]:2 = hlfir.declare %[[arg_temp_ptr]]
  ! CHECK: %[[ptr_comp:.*]] = hlfir.designate %[[temp]]#0{"ptr"}   {fortran_attrs = #fir.var_attrs<pointer>}
  ! CHECK: %[[ptr:.*]] = fir.load %[[ptr_comp]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: fir.store %[[ptr]] to %[[temp_ptr]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  temp_ptr => temp%ptr
end subroutine

! -----------------------------------------------------------------------------
!    Test pointer assignments with bound specs to contiguous right-hand side
! -----------------------------------------------------------------------------

! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from bound spec if specified
! CHECK-LABEL: func @_QPtest_array_with_new_lbs(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
subroutine test_array_with_new_lbs(p, x)
  real, target :: x(51:150)
  real, pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[shift:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
  ! CHECK: %[[box1:.*]] = fir.rebox %{{.*}}(%[[shift]]) : (!fir.box<!fir.array<100xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[c4:.*]] = fir.convert %c4{{.*}} : (i64) -> index
  ! CHECK: %[[shift2:.*]] = fir.shift %[[c4]] : (index) -> !fir.shift<1>
  ! CHECK: %[[box2:.*]] = fir.rebox %[[box1]](%[[shift2]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box2]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p(4:) => x
end subroutine

! Test F2018 10.2.2.3 point 9: bounds remapping
! CHECK-LABEL: func @_QPtest_array_remap(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.ref<!fir.array<100xf32>> {{{.*}}, fir.target})
subroutine test_array_remap(p, x)
  real, target :: x(100)
  real, pointer :: p(:, :)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[c2_idx:.*]] = fir.convert %c2{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[c11_idx:.*]] = fir.convert %c11{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[diff0:.*]] = arith.subi %[[c11_idx]], %[[c2_idx]] : index
  ! CHECK-DAG: %[[raw_ext0:.*]] = arith.addi %[[diff0:.*]], %c1{{.*}} : index
  ! CHECK-DAG: %[[cmp0:.*]] = arith.cmpi sgt, %[[raw_ext0]], %c0{{.*}} : index
  ! CHECK-DAG: %[[ext0:.*]] = arith.select %[[cmp0]], %[[raw_ext0]], %c0{{.*}} : index
  ! CHECK-DAG: %[[c3_idx:.*]] = fir.convert %c3{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[c12_idx:.*]] = fir.convert %c12{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[diff1:.*]] = arith.subi %[[c12_idx]], %[[c3_idx]] : index
  ! CHECK-DAG: %[[raw_ext1:.*]] = arith.addi %[[diff1]], %c1{{.*}} : index
  ! CHECK-DAG: %[[cmp1:.*]] = arith.cmpi sgt, %[[raw_ext1]], %c0{{.*}} : index
  ! CHECK-DAG: %[[ext1:.*]] = arith.select %[[cmp1]], %[[raw_ext1]], %c0{{.*}} : index
  ! CHECK: %[[shape:.*]] = fir.shape_shift %{{.*}}, %[[ext0]], %{{.*}}, %[[ext1]]
  ! CHECK: %[[box:.*]] = fir.rebox %{{.*}}(%[[shape]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_char_remap(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.boxchar<1> {{{.*}}, fir.target})
subroutine test_array_char_remap(p, x)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[arg1]]
  character(*), target :: x(100)
  character(:), pointer :: p(:, :)
  ! CHECK: subi
  ! CHECK: %[[ext0:.*]] = arith.select
  ! CHECK: subi
  ! CHECK: %[[ext1:.*]] = arith.select
  ! CHECK: %[[shape:.*]] = fir.shape_shift %{{.*}}, %[[ext0]], %{{.*}}, %[[ext1]]
  ! CHECK: %[[box:.*]] = fir.rebox %{{.*}}(%[[shape]]) : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK: fir.store %[[box]] to %[[p]]#0
  p(2:11, 3:12) => x
end subroutine

! -----------------------------------------------------------------------------
!  Test simple pointer assignments to non contiguous right-hand side
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_array_non_contig_rhs(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_rhs(p, x)
  real, target :: x(:)
  real, pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from rhs if no bounds spec.
! CHECK-LABEL: func @_QPtest_array_non_contig_rhs_lbs(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_rhs_lbs(p, x)
  real, target :: x(7:)
  real, pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[c7_idx:.*]] = fir.convert %c7{{.*}} : (i64) -> index
  ! CHECK: %[[shift:.*]] = fir.shift %[[c7_idx]] : (index) -> !fir.shift<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]](%[[shift]])
  ! CHECK: %[[shift2:.*]] = fir.shift %[[c7_idx]] : (index) -> !fir.shift<1>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]]#0(%[[shift2]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_non_contig_rhs2(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.ref<!fir.array<200xf32>> {{{.*}}, fir.target}) {
! CHECK:         %[[p:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:         %[[c200:.*]] = arith.constant 200 : index
! CHECK:         %[[shape_orig:.*]] = fir.shape %[[c200]] : (index) -> !fir.shape<1>
! CHECK:         %[[x:.*]]:2 = hlfir.declare %[[arg1]](%[[shape_orig]])
! CHECK:         %[[c10:.*]] = arith.constant 10 : index
! CHECK:         %[[c160:.*]] = arith.constant 160 : index
! CHECK:         %[[c3:.*]] = arith.constant 3 : index
! CHECK:         %[[c51:.*]] = arith.constant 51 : index
! CHECK:         %[[shape:.*]] = fir.shape %[[c51]] : (index) -> !fir.shape<1>
! CHECK:         %[[slice:.*]] = hlfir.designate %[[x]]#0 (%[[c10]]:%[[c160]]:%[[c3]])  shape %[[shape]] : (!fir.ref<!fir.array<200xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<51xf32>>
! CHECK:         %[[rebox:.*]] = fir.rebox %[[slice]] : (!fir.box<!fir.array<51xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         fir.store %[[rebox]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
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
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_rhs_new_lbs(p, x)
  real, target :: x(7:)
  real, pointer :: p(:)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[shift:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]](%[[shift]])
  ! CHECK: %[[shift1:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
  ! CHECK: %[[rebox1:.*]] = fir.rebox %[[x]]#0(%[[shift1]]) : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[shift2:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
  ! CHECK: %[[rebox2:.*]] = fir.rebox %[[rebox1]](%[[shift2]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>

  ! CHECK: fir.store %[[rebox2]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p(4:) => x
end subroutine

! Test F2018 10.2.2.3 point 9: bounds remapping
! CHECK-LABEL: func @_QPtest_array_non_contig_remap(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.box<!fir.array<?xf32>> {{{.*}}, fir.target})
subroutine test_array_non_contig_remap(p, x)
  real, target :: x(:)
  real, pointer :: p(:, :)
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK: %[[rebox1:.*]] = fir.rebox %[[x]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: subi
  ! CHECK: %[[ext0:.*]] = arith.select
  ! CHECK: subi
  ! CHECK: %[[ext1:.*]] = arith.select
  ! CHECK: %[[shape:.*]] = fir.shape_shift %{{.*}}, %[[ext0]], %{{.*}}, %[[ext1]]
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[rebox1]](%[[shape]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x
end subroutine

! Test remapping a slice

! CHECK-LABEL: func @_QPtest_array_non_contig_remap_slice(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>{{.*}}, %[[arg1:[^:]*]]: !fir.ref<!fir.array<400xf32>> {{{.*}}, fir.target}) {
! CHECK:         %[[p:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK:         %[[c400:.*]] = arith.constant 400 : index
! CHECK:         %[[shape400:.*]] = fir.shape %[[c400]] : (index) -> !fir.shape<1>
! CHECK:         %[[x:.*]]:2 = hlfir.declare %[[arg1]](%[[shape400]])
! CHECK:         %[[c51:.*]] = arith.constant 51 : index
! CHECK:         %[[c350:.*]] = arith.constant 350 : index
! CHECK:         %[[c3:.*]] = arith.constant 3 : index
! CHECK:         %[[c100:.*]] = arith.constant 100 : index
! CHECK:         %[[shape:.*]] = fir.shape %[[c100]] : (index) -> !fir.shape<1>
! CHECK:         %[[slice:.*]] = hlfir.designate %[[x]]#0 (%[[c51]]:%[[c350]]:%[[c3]])  shape %[[shape]] : (!fir.ref<!fir.array<400xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
! CHECK:         %[[rebox1:.*]] = fir.rebox %[[slice]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         subi
! CHECK:         %[[ext0:.*]] = arith.select
! CHECK:         subi
! CHECK:         %[[ext1:.*]] = arith.select
! CHECK:         %[[shape2:.*]] = fir.shape_shift %{{.*}}, %[[ext0]], %{{.*}}, %[[ext1]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[rebox2:.*]] = fir.rebox %[[rebox1]](%[[shape2]]) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[rebox2]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:         return
! CHECK:       }
subroutine test_array_non_contig_remap_slice(p, x)
  real, target :: x(400)
  real, pointer :: p(:, :)
  p(2:11, 3:12) => x(51:350:3)
end subroutine

! -----------------------------------------------------------------------------
!  Test pointer assignments where pointers are stored as descriptors (boxes).
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPissue857(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>>
subroutine issue857(rhs)
  type t
    integer :: i
  end type
  type(t), pointer :: rhs, lhs
  ! CHECK: %[[lhs_box:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>> {bindc_name = "lhs", uniq_name = "_QFissue857Elhs"}
  ! CHECK: %[[lhs:.*]]:2 = hlfir.declare %[[lhs_box]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFissue857Elhs"}
  ! CHECK: %[[rhs:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[box_load:.*]] = fir.load %[[rhs]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>>
  ! CHECK: fir.store %[[box_load]] to %[[lhs]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFissue857Tt{i:i32}>>>>
  lhs => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_array(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>>
subroutine issue857_array(rhs)
  type t
    integer :: i
  end type
  type(t), contiguous,  pointer :: rhs(:), lhs(:)
  ! CHECK: %[[lhs_box:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>> {bindc_name = "lhs", uniq_name = "_QFissue857_arrayElhs"}
  ! CHECK: %[[lhs:.*]]:2 = hlfir.declare %[[lhs_box]] {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFissue857_arrayElhs"}
  ! CHECK: %[[rhs:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[box_load:.*]] = fir.load %[[rhs]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_arrayTt{i:i32}>>>>>
  ! CHECK: fir.store %[[box_load]] to %[[lhs]]#0
  lhs => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_array_shift(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_shiftTt{i:i32}>>>>>
subroutine issue857_array_shift(rhs)
  ! Test lower bounds is the one from the shift
  type t
    integer :: i
  end type
  type(t), contiguous,  pointer :: rhs(:), lhs(:)
  ! CHECK: %[[lhs:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFissue857_array_shiftElhs"}
  ! CHECK: %[[rhs:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[box_load:.*]] = fir.load %[[rhs]]#0
  ! CHECK: %[[c42:.*]] = fir.convert %c42{{.*}} : (i64) -> index
  ! CHECK: %[[shift:.*]] = fir.shift %[[c42]] : (index) -> !fir.shift<1>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box_load]](%[[shift]]) : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_shiftTt{i:i32}>>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_shiftTt{i:i32}>>>>
  ! CHECK: fir.store %[[rebox]] to %[[lhs]]#0
  lhs(42:) => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_array_remap
subroutine issue857_array_remap(rhs)
  ! Test lower bounds is the one from the shift
  type t
    integer :: i
  end type
  type(t), contiguous,  pointer :: rhs(:, :), lhs(:)
  ! CHECK: %[[lhs:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFissue857_array_remapElhs"}
  ! CHECK: %[[rhs:.*]]:2 = hlfir.declare %{{.*}}
  ! CHECK: %[[box_load:.*]] = fir.load %[[rhs]]#0
  ! CHECK: %[[c101:.*]] = fir.convert %c101{{.*}} : (i64) -> index
  ! CHECK: %[[c200:.*]] = fir.convert %c200{{.*}} : (i64) -> index
  ! CHECK: %[[sub:.*]] = arith.subi %[[c200]], %[[c101]] : index
  ! CHECK: %[[raw_extent:.*]] = arith.addi %[[sub]], %c1{{.*}} : index
  ! CHECK: %[[cmp:.*]] = arith.cmpi sgt, %[[raw_extent]], %c0{{.*}} : index
  ! CHECK: %[[extent:.*]] = arith.select %[[cmp]], %[[raw_extent]], %c0{{.*}} : index
  ! CHECK: %[[ss:.*]] = fir.shape_shift %[[c101]], %[[extent]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box_load]](%[[ss]]) : (!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFissue857_array_remapTt{i:i32}>>>>
  ! CHECK: fir.store %[[rebox]] to %[[lhs]]#0
  lhs(101:200) => rhs
end subroutine

! CHECK-LABEL: func @_QPissue857_char
subroutine issue857_char(rhs)
  ! Check that the character slice is correctly reboxed into the pointer descriptor.
  character(:), contiguous,  pointer ::  lhs1(:), lhs2(:, :)
  character(*), target ::  rhs(100)
  ! CHECK: %[[lhs1:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFissue857_charElhs1"}
  ! CHECK: %[[lhs2:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "_QFissue857_charElhs2"}
  ! CHECK: %[[slice1:.*]] = hlfir.designate %{{.*}} ({{.*}}:{{.*}}:{{.*}})  shape %{{.*}} typeparams %{{.*}} : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<50x!fir.char<1,?>>>
  ! CHECK: %[[rebox1:.*]] = fir.rebox %[[slice1]] : (!fir.box<!fir.array<50x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.store %[[rebox1]] to %[[lhs1]]#0
  lhs1 => rhs(1:50:1)
  ! CHECK: %[[slice2:.*]] = hlfir.designate %{{.*}} ({{.*}}:{{.*}}:{{.*}})  shape %{{.*}} typeparams %{{.*}} : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<50x!fir.char<1,?>>>
  ! CHECK: %[[rebox2_src:.*]] = fir.rebox %[[slice2]] : (!fir.box<!fir.array<50x!fir.char<1,?>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: %[[rebox2:.*]] = fir.rebox %[[rebox2_src]](%{{.*}}) : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK: fir.store %[[rebox2]] to %[[lhs2]]#0
  lhs2(1:2, 1:25) => rhs(1:50:1)
end subroutine

! CHECK-LABEL: func @_QPissue1180(
! CHECK-SAME:  %[[arg0:[^:]*]]: !fir.ref<i32> {{{.*}}, fir.target}) {
subroutine issue1180(x)
  integer, target :: x
  integer, pointer :: p
  common /some_common/ p
  ! CHECK: %[[VAL_1:.*]] = fir.address_of(@some_common_) : !fir.ref<!fir.array<24xi8>>
  ! CHECK: %[[VAL_3:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_3]] : (!fir.ref<!fir.array<24xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK: %[[p:.*]]:2 = hlfir.declare %[[VAL_5]] storage
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[VAL_6:.*]] = fir.embox %[[x]]#0 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  ! CHECK: fir.store %[[VAL_6]] to %[[p]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  p => x
end subroutine
