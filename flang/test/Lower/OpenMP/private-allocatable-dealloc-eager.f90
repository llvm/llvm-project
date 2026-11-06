! RUN: bbc -fopenmp -emit-hlfir --enable-delayed-privatization=false %s -o - | FileCheck %s --implicit-check-not=fir.freemem --implicit-check-not=omp.private
! RUN: %flang_fc1 -fopenmp -emit-hlfir -mmlir --enable-delayed-privatization=false %s -o - | FileCheck %s --implicit-check-not=fir.freemem --implicit-check-not=omp.private

! Restricting cleanup to locally privatized symbols must still work with eager
! privatization. The local allocatable must be freed exactly once after sections.
! CHECK-LABEL: func.func @_QPlocal_cleanup(
subroutine local_cleanup(a, p)
  integer, allocatable :: a
  integer, pointer :: p
  integer :: i
  ! CHECK: %[[A_STORAGE:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
  ! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_STORAGE]] uniq_name("_QFlocal_cleanupEa")
  ! CHECK: omp.sections {
  ! CHECK-NEXT: omp.section {
  !$omp sections private(a, p, i)
    !$omp section
    a = 42
    i = 1
    nullify(p)
  ! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: }
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: }
  !$omp end sections
  ! CHECK-NEXT: %[[BOX:.*]] = fir.load %[[A]]#0
  ! CHECK-NEXT: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
  ! CHECK-NEXT: %[[INT:.*]] = fir.convert %[[ADDR]]
  ! CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
  ! CHECK-NEXT: %[[ALLOCATED:.*]] = arith.cmpi ne, %[[INT]], %[[ZERO]]
  ! CHECK-NEXT: fir.if %[[ALLOCATED]] {
  ! CHECK-NEXT: %[[FREE_BOX:.*]] = fir.load %[[A]]#0
  ! CHECK-NEXT: %[[FREE_ADDR:.*]] = fir.box_addr %[[FREE_BOX]]
  ! CHECK-NEXT: fir.freemem %[[FREE_ADDR]] : !fir.heap<i32>
  ! CHECK: fir.store %{{.*}} to %[[A]]#0
  ! CHECK-NEXT: }
  ! CHECK-NEXT: return
end subroutine
