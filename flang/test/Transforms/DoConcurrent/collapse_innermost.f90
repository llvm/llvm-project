! Tests the number of loops collapsed when a multi-index `do concurrent` loop
! is mapped to OpenMP. On the host, a vectorizable innermost loop is not
! collapsed. On the device, all loops are always collapsed.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,HOST
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s --check-prefixes=CHECK,DEVICE

! The innermost index is the first subscript of every access.
! CHECK-LABEL: func.func @_QPunit_stride
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(3) {
subroutine unit_stride(s, x, nr, nt, np)
  integer :: nr, nt, np, i, j, k
  real(8) :: s(nr, nt, np), x(nr, nt, np)
  do concurrent (k=1:np, j=1:nt, i=1:nr)
    s(i, j, k) = x(i, j, k)
  end do
end subroutine

! Offsets on other indices do not matter.
! CHECK-LABEL: func.func @_QPother_offsets
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine other_offsets(s, x, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n)
  do concurrent (j=2:n, i=1:n)
    s(i, j) = x(i, j - 1)
  end do
end subroutine

! Subscripts computed from the innermost index are not inspected; the plain
! `s(i, j)` access is enough.
! CHECK-LABEL: func.func @_QPstencil
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine stencil(s, x, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n)
  do concurrent (j=1:n, i=2:n - 1)
    s(i, j) = x(i - 1, j) + x(i + 1, j)
  end do
end subroutine

! The innermost index is the last subscript.
! CHECK-LABEL: func.func @_QPstrided
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(3) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(3) {
subroutine strided(s, x, nr, nt, np)
  integer :: nr, nt, np, i, j, k
  real(8) :: s(nr, nt, np), x(nr, nt, np)
  do concurrent (i=1:nr, j=1:nt, k=1:np)
    s(i, j, k) = x(i, j, k)
  end do
end subroutine

! A contiguous store mixed with a strided load.
! CHECK-LABEL: func.func @_QPtransposed_copy
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine transposed_copy(s, x, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = x(j, i)
  end do
end subroutine

! A call in the body.
! CHECK-LABEL: func.func @_QPwith_call
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine with_call(s, n)
  integer :: n, i, j
  real(8) :: s(n, n)
  interface
    pure real(8) function f(i)
      integer, intent(in) :: i
    end function
  end interface
  do concurrent (j=1:n, i=1:n)
    s(i, j) = f(i)
  end do
end subroutine

! Assumed-shape arrays are not known to be contiguous.
! CHECK-LABEL: func.func @_QPassumed_shape
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine assumed_shape(s, x)
  integer :: i, j
  real(8) :: s(:, :), x(:, :)
  do concurrent (j=1:size(s, 2), i=1:size(s, 1))
    s(i, j) = x(i, j)
  end do
end subroutine

! Assumed-shape arrays with the contiguous attribute.
! CHECK-LABEL: func.func @_QPassumed_shape_contiguous
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine assumed_shape_contiguous(s, x)
  integer :: i, j
  real(8), contiguous :: s(:, :), x(:, :)
  do concurrent (j=1:size(s, 2), i=1:size(s, 1))
    s(i, j) = x(i, j)
  end do
end subroutine

! Allocatable arrays are contiguous.
! CHECK-LABEL: func.func @_QPallocatable
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine allocatable(s, x, n)
  integer :: n, i, j
  real(8), allocatable :: s(:, :), x(:, :)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = x(i, j)
  end do
end subroutine

! A conditional in the body.
! CHECK-LABEL: func.func @_QPwith_if
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine with_if(s, x, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n)
  do concurrent (j=1:n, i=1:n)
    if (x(i, j) > 0) s(i, j) = x(i, j)
  end do
end subroutine

! An array assignment in the body.
! CHECK-LABEL: func.func @_QParray_assign
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine array_assign(s, n)
  integer :: n, i, j
  real(8) :: s(4, n, n)
  do concurrent (j=1:n, i=1:n)
    s(:, i, j) = 0
  end do
end subroutine

! The innermost index is the first index of a strided array section.
! CHECK-LABEL: func.func @_QProw_section
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine row_section(s, x, y, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n), y(n)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = dot_product(x(i, :), y)
  end do
end subroutine

! A transformational intrinsic on an array section not indexed by the
! innermost index.
! CHECK-LABEL: func.func @_QPcolumn_sum
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine column_sum(s, x, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = x(i, j) + sum(x(:, j))
  end do
end subroutine

! A transformational intrinsic on a whole array.
! CHECK-LABEL: func.func @_QPwhole_array_sum
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine whole_array_sum(s, x, y, n)
  integer :: n, i, j
  real(8) :: s(n, n), x(n, n), y(n)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = x(i, j) + sum(y)
  end do
end subroutine

! The outer loop alone is too short to keep the threads busy.
! CHECK-LABEL: func.func @_QPfew_outer
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine few_outer(s, n)
  integer :: n, i, j
  real(8) :: s(n, 4)
  do concurrent (j=1:4, i=1:n)
    s(i, j) = 0
  end do
end subroutine

! The outer loop runs exactly the minimum number of iterations.
! CHECK-LABEL: func.func @_QPenough_outer
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine enough_outer(s, n)
  integer :: n, i, j
  real(8) :: s(n, 64)
  do concurrent (j=1:64, i=1:n)
    s(i, j) = 0
  end do
end subroutine

! Short outer loops whose combined iterations are enough.
! CHECK-LABEL: func.func @_QPenough_outer_product
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(3) {
subroutine enough_outer_product(s, n)
  integer :: n, i, j, k
  real(8) :: s(n, 8, 8)
  do concurrent (k=1:8, j=1:8, i=1:n)
    s(i, j, k) = 0
  end do
end subroutine

! A component of an array of derived type is a strided access.
! CHECK-LABEL: func.func @_QPderived_component
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine derived_component(s, a, n)
  type t
    real(8) :: x, y
  end type
  integer :: n, i, j
  type(t) :: a(n, n)
  real(8) :: s(n, n)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = a(i, j)%x
  end do
end subroutine

! An element of an array component of a scalar is contiguous.
! CHECK-LABEL: func.func @_QParray_component
! HOST: omp.loop_nest {{.*}} step ({{[^)]*}}) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine array_component(s, a, n)
  type t
    real(8) :: x(100)
  end type
  integer :: n, i, j
  type(t) :: a
  real(8) :: s(100, n)
  do concurrent (j=1:n, i=1:100)
    s(i, j) = a%x(i)
  end do
end subroutine

! A complex part is a strided access.
! CHECK-LABEL: func.func @_QPcomplex_part
! HOST: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
! DEVICE: omp.loop_nest {{.*}} step ({{.*}}) collapse(2) {
subroutine complex_part(s, z, n)
  integer :: n, i, j
  complex(8) :: z(n, n)
  real(8) :: s(n, n)
  do concurrent (j=1:n, i=1:n)
    s(i, j) = z(i, j)%re
  end do
end subroutine
