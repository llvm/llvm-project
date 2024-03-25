! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: bbc -emit-fir -hlfir=false -gen-array-coor %s -o - | FileCheck %s --check-prefix=ArrayCoorCHECK

! Test that non-contiguous assumed-shape memory layout is handled in lowering.
! In practice, test that input fir.box is propagated to fir operations 

! Also test that when the contiguous keyword is present, lowering adds the
! attribute to the fir argument and that is takes the contiguity into account
! In practice, test that the input fir.box is not propagated to fir operations.

! CHECK-LABEL: func @_QPtest_element_ref(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.box<!fir.array<?xf32>>{{.*}}) {
! ArrayCoorCHECK-LABEL: func @_QPtest_element_ref
subroutine test_element_ref(x, y)
  real, contiguous :: x(:)
  ! CHECK-DAG: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real :: y(4:)
  ! CHECK-DAG: %[[c4:.*]] = fir.convert %c4{{.*}} : (i64) -> index

  call bar(x(100))
  ! CHECK: fir.coordinate_of %[[xaddr]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  call bar(y(100))
  ! Test that for an entity that is not know to be contiguous, the fir.box is passed
  ! to coordinate of and that the lower bounds is already applied by lowering.
  ! CHECK: %[[c4_2:.*]] = fir.convert %[[c4]] : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c100{{.*}}, %[[c4_2]] : i64
  ! CHECK: fir.coordinate_of %arg1, %{{.*}} : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>


  ! Repeat test when lowering is using fir.array_coor
  ! ArrayCoorCHECK-DAG: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! ArrayCoorCHECK-DAG: %[[xshape:.*]] = fir.shape
  ! ArrayCoorCHECK-DAG: %[[c100:.*]] = fir.convert %c100{{.*}} : (i64) -> index
  ! ArrayCoorCHECK: fir.array_coor %[[xaddr]](%[[xshape]]) %[[c100]] : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>

  ! ArrayCoorCHECK-DAG: %[[c100_1:.*]] = fir.convert %c100{{.*}} : (i64) -> index
  ! ArrayCoorCHECK-DAG: %[[shift:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
  ! ArrayCoorCHECK: fir.array_coor %arg1(%[[shift]]) %[[c100_1]] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
end subroutine

! CHECK-LABEL: func @_QPtest_element_assign(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.box<!fir.array<?xf32>>{{.*}}) {
!  ArrayCoorCHECK-LABEL: func @_QPtest_element_assign
subroutine test_element_assign(x, y)
  real, contiguous :: x(:)
  ! CHECK-DAG: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real :: y(4:)
  ! CHECK-DAG: %[[c4:.*]] = fir.convert %c4{{.*}} : (i64) -> index
  x(100) = 42.
  ! CHECK: fir.coordinate_of %[[xaddr]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
  y(100) = 42.
  ! CHECK: %[[c4_2:.*]] = fir.convert %[[c4]] : (index) -> i64
  ! CHECK: %[[index:.*]] = arith.subi %c100{{.*}}, %[[c4_2]] : i64
  ! CHECK: fir.coordinate_of %arg1, %{{.*}} : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>

  ! ArrayCoorCHECK-DAG: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  ! ArrayCoorCHECK-DAG: %[[xshape:.*]] = fir.shape
  ! ArrayCoorCHECK-DAG: %[[c100:.*]] = fir.convert %c100{{.*}} : (i64) -> index
  ! ArrayCoorCHECK: fir.array_coor %[[xaddr]](%[[xshape]]) %[[c100]] : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>

  ! ArrayCoorCHECK-DAG: %[[c100_1:.*]] = fir.convert %c100{{.*}} : (i64) -> index
  ! ArrayCoorCHECK-DAG: %[[shift:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
  ! ArrayCoorCHECK: fir.array_coor %arg1(%[[shift]]) %[[c100_1]] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
end subroutine

! CHECK-LABEL: func @_QPtest_ref_in_array_expr(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_ref_in_array_expr(x, y)
  real, contiguous :: x(:)
  ! CHECK: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real :: y(:)
  call bar2(x+1.)
  ! CHECK: fir.array_load %[[xaddr]](%{{.*}}) : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  call bar2(y+1.)
  ! CHECK: fir.array_load %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
end subroutine


! CHECK-LABEL: func @_QPtest_assign_in_array_ref(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.box<!fir.array<?xf32>>{{.*}}) {
subroutine test_assign_in_array_ref(x, y)
  real, contiguous :: x(:)
  ! CHECK: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real :: y(:)
  x = 42.
  ! CHECK: %[[xload:.*]] = fir.array_load %[[xaddr]]({{.*}}) : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK: %[[xloop:.*]] = fir.do_loop {{.*}} iter_args(%arg3 = %[[xload]]) -> (!fir.array<?xf32>)
  ! CHECK: fir.array_merge_store %[[xload]], %[[xloop]] to %[[xaddr]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>
  y = 42.
  ! CHECK: %[[yload:.*]] = fir.array_load %arg1 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[yloop:.*]] = fir.do_loop {{.*}} iter_args(%arg3 = %[[yload]]) -> (!fir.array<?xf32>) {
  ! CHECK: fir.array_merge_store %[[yload]], %[[yloop]] to %arg1 : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
end subroutine

! CHECK-LABEL: func @_QPtest_slice_ref(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.box<!fir.array<?xf32>>
subroutine test_slice_ref(x, y, z1, z2, i, j, k, n)
  real, contiguous :: x(:)
  ! CHECK: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real :: y(:)
  integer :: i, j, k, n
  real :: z1(n), z2(n)
  z2 = x(i:j:k)
  ! CHECK: %[[xslice:.*]] = fir.slice
  ! CHECK: fir.array_load %[[xaddr]]{{.*}}%[[xslice]]{{.*}}: (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  z1 = y(i:j:k)
  ! CHECK: %[[yslice:.*]] = fir.slice
  ! CHECK: fir.array_load %arg1 {{.*}}%[[yslice]]{{.*}} : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
end subroutine

! CHECK-LABEL: func @_QPtest_slice_assign(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.contiguous}, %arg1: !fir.box<!fir.array<?xf32>>
subroutine test_slice_assign(x, y, i, j, k)
  real, contiguous :: x(:)
  ! CHECK: %[[xaddr:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  real :: y(:)
  integer :: i, j, k
  x(i:j:k) = 42.
  ! CHECK: %[[xslice:.*]] = fir.slice
  ! CHECK: fir.array_load %[[xaddr]]{{.*}}%[[xslice]]{{.*}}: (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  y(i:j:k) = 42.
  ! CHECK: %[[yslice:.*]] = fir.slice
  ! CHECK: fir.array_load %arg1 {{.*}}%[[yslice]]{{.*}}: (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
end subroutine

! test that allocatable are considered contiguous.
! CHECK-LABEL: func @_QPfoo
subroutine foo(x)
  real, allocatable :: x(:)
  call bar(x(100))
  ! CHECK: fir.coordinate_of %{{.*}}, %{{.*}} (!fir.heap<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
end subroutine

! Test that non-contiguous dummy are propagated with their memory layout (we
! mainly do not want to create a new box that would ignore the original layout).
! CHECK: func @_QPpropagate(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"})
subroutine propagate(x)
  interface
    subroutine bar3(x)
      real :: x(:)
    end subroutine
  end interface
  real :: x(:)
  call bar3(x)
 ! CHECK: fir.call @_QPbar3(%arg0) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
end subroutine
