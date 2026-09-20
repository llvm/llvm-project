! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test that non-contiguous assumed-shape memory layout is handled in lowering.
! In practice, test that input fir.box is propagated to fir operations

! Also test that when the contiguous keyword is present, lowering adds the
! attribute to the fir argument and that is takes the contiguity into account
! In practice, test that the input fir.box is not propagated to fir operations.

! CHECK-LABEL: func @_QPtest_element_ref(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
subroutine test_element_ref(x, y)
  real, contiguous :: x(:)
  ! CHECK: %[[X_REF:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]({{.*}}) {{.*}}uniq_name = "_QFtest_element_refEx"{{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  real :: y(4:)
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]]({{.*}}) {{.*}}uniq_name = "_QFtest_element_refEy"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)

  call bar(x(100))
  ! CHECK: %[[X_ELT:.*]] = hlfir.designate %[[X_DECL]]#0 (%c100)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPbar(%[[X_ELT]]) {{.*}} : (!fir.ref<f32>) -> ()
  call bar(y(100))
  ! CHECK: %[[Y_ELT:.*]] = hlfir.designate %[[Y_DECL]]#0 (%c100_0)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPbar(%[[Y_ELT]]) {{.*}} : (!fir.ref<f32>) -> ()
end subroutine

! CHECK-LABEL: func @_QPtest_element_assign(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
subroutine test_element_assign(x, y)
  real, contiguous :: x(:)
  ! CHECK: %[[X_REF:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]({{.*}}) {{.*}}uniq_name = "_QFtest_element_assignEx"{{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  real :: y(4:)
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]]({{.*}}) {{.*}}uniq_name = "_QFtest_element_assignEy"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  x(100) = 42.
  ! CHECK: %[[X_ELT:.*]] = hlfir.designate %[[X_DECL]]#0 (%c100)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK: hlfir.assign %cst to %[[X_ELT]] : f32, !fir.ref<f32>
  y(100) = 42.
  ! CHECK: %[[Y_ELT:.*]] = hlfir.designate %[[Y_DECL]]#0 (%c100_1)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK: hlfir.assign %cst_0 to %[[Y_ELT]] : f32, !fir.ref<f32>
end subroutine

! CHECK-LABEL: func @_QPtest_ref_in_array_expr(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
subroutine test_ref_in_array_expr(x, y)
  real, contiguous :: x(:)
  ! CHECK: %[[X_REF:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]({{.*}}) {{.*}}uniq_name = "_QFtest_ref_in_array_exprEx"{{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  real :: y(:)
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]] {{.*}}uniq_name = "_QFtest_ref_in_array_exprEy"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  call bar2(x+1.)
  ! CHECK: hlfir.elemental {{.*}}
  ! CHECK: hlfir.designate %[[X_DECL]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  call bar2(y+1.)
  ! CHECK: hlfir.elemental {{.*}}
  ! CHECK: hlfir.designate %[[Y_DECL]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
end subroutine


! CHECK-LABEL: func @_QPtest_assign_in_array_ref(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
subroutine test_assign_in_array_ref(x, y)
  real, contiguous :: x(:)
  ! CHECK: %[[X_REF:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]({{.*}}) {{.*}}uniq_name = "_QFtest_assign_in_array_refEx"{{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  real :: y(:)
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]] {{.*}}uniq_name = "_QFtest_assign_in_array_refEy"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  x = 42.
  ! CHECK: hlfir.assign %cst to %[[X_DECL]]#0 : f32, !fir.box<!fir.array<?xf32>>
  y = 42.
  ! CHECK: hlfir.assign %cst_0 to %[[Y_DECL]]#0 : f32, !fir.box<!fir.array<?xf32>>
end subroutine

! CHECK-LABEL: func @_QPtest_slice_ref(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"},
subroutine test_slice_ref(x, y, z1, z2, i, j, k, n)
  real, contiguous :: x(:)
  ! CHECK: %[[X_REF:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]({{.*}}) {{.*}}uniq_name = "_QFtest_slice_refEx"{{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  real :: y(:)
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]] {{.*}}uniq_name = "_QFtest_slice_refEy"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  integer :: i, j, k, n
  real :: z1(n), z2(n)
  z2 = x(i:j:k)
  ! CHECK: %[[X_SLICE:.*]] = hlfir.designate %[[X_DECL]]#0 {{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: hlfir.assign %[[X_SLICE]] to %{{.*}} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
  z1 = y(i:j:k)
  ! CHECK: %[[Y_SLICE:.*]] = hlfir.designate %[[Y_DECL]]#0 {{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: hlfir.assign %[[Y_SLICE]] to %{{.*}} : !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>
end subroutine

! CHECK-LABEL: func @_QPtest_slice_assign(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"},
subroutine test_slice_assign(x, y, i, j, k)
  real, contiguous :: x(:)
  ! CHECK: %[[X_REF:.*]] = fir.box_addr %[[ARG0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_REF]]({{.*}}) {{.*}}uniq_name = "_QFtest_slice_assignEx"{{.*}} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  real :: y(:)
  ! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]] {{.*}}uniq_name = "_QFtest_slice_assignEy"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  integer :: i, j, k
  x(i:j:k) = 42.
  ! CHECK: %[[X_SLICE:.*]] = hlfir.designate %[[X_DECL]]#0 {{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: hlfir.assign %cst to %[[X_SLICE]] : f32, !fir.box<!fir.array<?xf32>>
  y(i:j:k) = 42.
  ! CHECK: %[[Y_SLICE:.*]] = hlfir.designate %[[Y_DECL]]#0 {{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: hlfir.assign %cst_1 to %[[Y_SLICE]] : f32, !fir.box<!fir.array<?xf32>>
end subroutine

! test that allocatable are considered contiguous.
! CHECK-LABEL: func @_QPfoo(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "x"}) {
subroutine foo(x)
  real, allocatable :: x(:)
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}}uniq_name = "_QFfooEx"{{.*}}
  call bar(x(100))
  ! CHECK: %[[X_LOAD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[X_ELT:.*]] = hlfir.designate %[[X_LOAD]] (%c100)  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
  ! CHECK: fir.call @_QPbar(%[[X_ELT]]) {{.*}} : (!fir.ref<f32>) -> ()
end subroutine

! Test that non-contiguous dummy are propagated with their memory layout (we
! mainly do not want to create a new box that would ignore the original layout).
! CHECK: func @_QPpropagate(%[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"})
subroutine propagate(x)
  interface
    subroutine bar3(x)
      real :: x(:)
    end subroutine
  end interface
  real :: x(:)
  ! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}}uniq_name = "_QFpropagateEx"{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  call bar3(x)
 ! CHECK: fir.call @_QPbar3(%[[X_DECL]]#0) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
end subroutine
