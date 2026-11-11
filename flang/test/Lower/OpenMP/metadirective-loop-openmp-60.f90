! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=60 %s -o - | \
! RUN:   FileCheck %s --implicit-check-not='linear('

! CHECK: omp.private {type = private} @[[PRIVATE:.*]] : i32

! CHECK-LABEL: func.func @_QPtest_metadirective_simd_v60(
! CHECK: %[[N:.*]]:2 = hlfir.declare {{.*}}metadirective_simd_v60En
! CHECK: %[[AFTER:.*]]:2 = hlfir.declare {{.*}}metadirective_simd_v60Eafter
! CHECK: %[[I:.*]]:2 = hlfir.declare {{.*}}metadirective_simd_v60Ei
! CHECK: omp.simd private(@[[PRIVATE]] %[[I]]#0 ->
! CHECK-SAME: %[[I_PRIV:[^ ]+]] : !fir.ref<i32>) {
! CHECK: omp.loop_nest (%[[IV:.*]]) : i32
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_PRIV]]
! CHECK: hlfir.assign %[[IV]] to %[[I_DECL]]#0
! CHECK: %[[VALUE:.*]] = fir.load %[[I_DECL]]#0
! CHECK: %[[INDEX:.*]] = fir.load %[[I_DECL]]#0
! CHECK: %[[INDEX_I64:.*]] = fir.convert %[[INDEX]]
! CHECK: %[[ELEMENT:.*]] = hlfir.designate
! CHECK-SAME: (%[[INDEX_I64]])
! CHECK: hlfir.assign %[[VALUE]] to %[[ELEMENT]]
! CHECK: %[[BOUND:.*]] = fir.load %[[N]]#0
! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
! CHECK: %[[NEXT:.*]] = arith.addi %[[IV]], %[[STEP]] : i32
! CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK: %[[NEG:.*]] = arith.cmpi slt, %[[STEP]], %[[ZERO]] : i32
! CHECK: %[[LT:.*]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
! CHECK: %[[GT:.*]] = arith.cmpi sgt, %[[NEXT]], %[[BOUND]] : i32
! CHECK: %[[LAST:.*]] = arith.select %[[NEG]], %[[LT]], %[[GT]] : i1
! CHECK: fir.if %[[LAST]] {
! CHECK-NOT: {{^ *}}}
! CHECK: hlfir.assign %[[NEXT]] to %[[I_DECL]]#0
! CHECK-NOT: {{^ *}}}
! CHECK: %[[COPY:.*]] = fir.load %[[I_DECL]]#0
! CHECK-NOT: {{^ *}}}
! CHECK: hlfir.assign %[[COPY]] to %[[I]]#0
! CHECK: omp.yield
! CHECK: %[[AFTER_VALUE:.*]] = fir.load %[[I]]#0
! CHECK-NEXT: hlfir.assign %[[AFTER_VALUE]] to %[[AFTER]]#0

subroutine test_metadirective_simd_v60(n, a, after)
  integer :: n, a(n), after, i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
end subroutine
