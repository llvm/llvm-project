! Test lowering of pointer assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s


! Note that p => NULL() are tested in pointer-disassociate.f90

! -----------------------------------------------------------------------------
!     Test simple pointer assignments to contiguous right-hand side
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_scalar(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>,
! CHECK-SAME: %[[x:.*]]: !fir.ref<f32> {fir.target})
subroutine test_scalar(p, x)
  real, target :: x
  real, pointer :: p
  ! CHECK: %[[box:.*]] = fir.embox %[[x]] : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_scalar_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.boxchar<1> {fir.target})
subroutine test_scalar_char(p, x)
  character(*), target :: x
  character(:), pointer :: p
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[box:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<100xf32>> {fir.target})
subroutine test_array(p, x)
  real, target :: x(100)
  real, pointer :: p(:)
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_char(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.boxchar<1> {fir.target}) {
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
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<100xf32>> {fir.target})
subroutine test_array_remap(p, x)
  real, target :: x(100)
  real, pointer :: p(:, :)
  ! CHECK-DAG: %[[c2_idx:.*]] = fir.convert %c2{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[c11_idx:.*]] = fir.convert %c11{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[diff0:.*]] = subi %[[c11_idx]], %[[c2_idx]] : index
  ! CHECK-DAG: %[[ext0:.*]] = addi %[[diff0:.*]], %c1{{.*}} : index
  ! CHECK-DAG: %[[c3_idx:.*]] = fir.convert %c3{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[c12_idx:.*]] = fir.convert %c12{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[diff1:.*]] = subi %[[c12_idx]], %[[c3_idx]] : index
  ! CHECK-DAG: %[[ext1:.*]] = addi %[[diff1]], %c1{{.*}} : index
  ! CHECK-DAG: %[[addrCast:.*]] = fir.convert %[[x]] : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c2{{.*}}, %[[ext0]], %c3{{.*}}, %[[ext1]]
  ! CHECK: %[[box:.*]] = fir.embox %[[addrCast]](%[[shape]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[box]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x
end subroutine

! CHECK-LABEL: func @_QPtest_array_char_remap(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.boxchar<1> {fir.target})
subroutine test_array_char_remap(p, x)
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[x]]
  character(*), target :: x(100)
  character(:), pointer :: p(:, :)
  ! CHECK: subi
  ! CHECK: %[[ext0:.*]] = addi
  ! CHECK: subi
  ! CHECK: %[[ext1:.*]] = addi
  ! CHECK: %[[shape:.*]] = fir.shape_shift %c2{{.*}}, %[[ext0]], %c3{{.*}}, %[[ext1]]
  ! CHECK: %[[box:.*]] = fir.embox %{{.*}}(%[[shape]]) typeparams %[[unbox]]#1 : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shapeshift<2>, index) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>
  ! CHECK: fir.store %[[box]] to %[[p]]
  p(2:11, 3:12) => x
end subroutine

! -----------------------------------------------------------------------------
!  Test simple pointer assignments to non contiguous right-hand side
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest_array_non_contig_rhs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>> {fir.target})
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
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>> {fir.target})
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
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<200xf32>> {fir.target})
subroutine test_array_non_contig_rhs2(p, x)
  real, target :: x(200)
  real, pointer :: p(:)
  ! CHECK: %[[shape:.*]] = fir.shape %c200{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[slice:.*]] = fir.slice %c10{{.*}}, %c160{{.*}}, %c3{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) {{.}}%[[slice]]{{.}} : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  p => x(10:160:3)
end subroutine

! -----------------------------------------------------------------------------
!  Test pointer assignments with bound specs to non contiguous right-hand side
! -----------------------------------------------------------------------------


! Test 10.2.2.3 point 10: lower bounds requirements:
! pointer takes lbounds from bound spec if specified
! CHECK-LABEL: func @_QPtest_array_non_contig_rhs_new_lbs(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>> {fir.target})
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
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>> {fir.target})
subroutine test_array_non_contig_remap(p, x)
  real, target :: x(:)
  real, pointer :: p(:, :)
  ! CHECK: subi
  ! CHECK: %[[ext0:.*]] = addi
  ! CHECK: subi
  ! CHECK: %[[ext1:.*]] = addi
  ! CHECK: %[[shape:.*]] = fir.shape_shift %{{.*}}, %[[ext0]], %{{.*}}, %[[ext1]]
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[x]](%[[shape]]) : (!fir.box<!fir.array<?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x
end subroutine

! Test remapping a slice
! CHECK-LABEL: func @_QPtest_array_non_contig_remap_slice(
! CHECK-SAME: %[[p:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>,
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<400xf32>> {fir.target})
subroutine test_array_non_contig_remap_slice(p, x)
  real, target :: x(400)
  real, pointer :: p(:, :)
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %c400{{.*}}
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %c51{{.*}}, %c350{{.*}}, %c3{{.*}}
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) {{.}}%[[slice]]{{.}} : (!fir.ref<!fir.array<400xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>

  ! CHECK: %[[reshape:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]](%[[reshape]]) : (!fir.box<!fir.array<?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[rebox]] to %[[p]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  p(2:11, 3:12) => x(51:350:3)
end subroutine
