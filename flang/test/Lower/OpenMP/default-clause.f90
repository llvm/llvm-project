! This test checks lowering of OpenMP parallel directive
! with `DEFAULT` clause present.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - \
! RUN: | FileCheck %s

! RUN: bbc -fopenmp -emit-hlfir %s -o - \
! RUN: | FileCheck %s


!CHECK: func @_QQmain() attributes {fir.bindc_name = "DEFAULT_CLAUSE_LOWERING"} {
!CHECK: %[[W:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFEw"}
!CHECK: %[[W_DECL:.*]]:2 = hlfir.declare %[[W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_W:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 2 : i32
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[RESULT:.*]] = arith.muli %[[CONST]], %[[TEMP]] : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_W_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 45 : i32
!CHECK: %[[RESULT:.*]] = arith.addi %[[TEMP]], %[[CONST]] : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

program default_clause_lowering
    integer :: x, y, z, w

    !$omp parallel default(private) firstprivate(x) shared(z)
        x = y * 2
        z = w + 45
    !$omp end parallel

!CHECK: omp.parallel {
!CHECK: %[[TEMP:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(shared)
        x = y
    !$omp end parallel

!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(none) private(x, y)
        x = y
    !$omp end parallel

!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(firstprivate) firstprivate(y)
        x = y
    !$omp end parallel

!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_W:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 2 : i32
!CHECK: %[[RESULT:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[TEMP:.*]] = arith.muli %[[CONST]], %[[RESULT]] : i32
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_W_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 45 : i32
!CHECK: %[[RESULT:.*]] = arith.addi %[[TEMP]], %[[CONST]] : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(firstprivate) private(x) shared(z)
        x = y * 2
        z = w + 45
    !$omp end parallel

!CHECK: omp.parallel   {
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_W:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_W_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
    !$omp parallel
        !$omp parallel default(private)
            x = y
        !$omp end parallel

        !$omp parallel default(firstprivate)
            w = x
        !$omp end parallel
    !$omp end parallel

end program default_clause_lowering

!CHECK-LABEL: func @_QPnested_default_clause_test1
!CHECK: %[[K:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFnested_default_clause_test1Ek"}
!CHECK: %[[K_DECL:.*]]:2 = hlfir.declare %[[K]] {uniq_name = "_QFnested_default_clause_test1Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[W:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFnested_default_clause_test1Ew"}
!CHECK: %[[W_DECL:.*]]:2 = hlfir.declare %[[W]] {uniq_name = "_QFnested_default_clause_test1Ew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFnested_default_clause_test1Ex"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFnested_default_clause_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFnested_default_clause_test1Ey"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFnested_default_clause_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFnested_default_clause_test1Ez"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFnested_default_clause_test1Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Z:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_K:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_test1Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_K_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_K]] {uniq_name = "_QFnested_default_clause_test1Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[INNER_PRIVATE_Y:.*]], {{.*}} {{.*}}#0 -> %[[INNER_PRIVATE_X:.*]] : {{.*}}) {
!CHECK: %[[INNER_PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_X]] {uniq_name = "_QFnested_default_clause_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 20 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_Y_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 10 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[INNER_PRIVATE_W:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_PRIVATE_Z:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_PRIVATE_K:.*]] : {{.*}}) {
!CHECK: %[[INNER_PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_W]] {uniq_name = "_QFnested_default_clause_test1Ew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_test1Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_K_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_K]] {uniq_name = "_QFnested_default_clause_test1Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 30 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[PRIVATE_Y_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 40 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_W_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 50 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 40 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_K_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
subroutine nested_default_clause_test1
  integer :: x, y, z, w, k

  !$omp parallel  firstprivate(x) private(y) shared(w) default(private)
    !$omp parallel default(private)
     y = 20
     x = 10
    !$omp end parallel

    !$omp parallel default(firstprivate) shared(y) private(w)
      y = 30
      w = 40
      z = 50
      k = 40
    !$omp end parallel
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPnested_default_clause_test2
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_W:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Z:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test2Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFnested_default_clause_test2Ew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_test2Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_INNER_X:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_PRIVATE_Y:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_PRIVATE_W:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_INNER_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_INNER_X]] {uniq_name = "_QFnested_default_clause_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test2Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_W]] {{.*}}
!CHECK: %[[TEMP:.*]] = fir.load %[[INNER_PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_INNER_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_INNER_W:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_INNER_X:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_INNER_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_INNER_W]] {uniq_name = "_QFnested_default_clause_test2Ew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_INNER_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_INNER_X]] {uniq_name = "_QFnested_default_clause_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP_1:.*]] = fir.load %[[PRIVATE_INNER_X_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[TEMP_2:.*]] = fir.load %[[PRIVATE_Z_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[RESULT:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[PRIVATE_INNER_W_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: }
subroutine nested_default_clause_test2
  integer :: x, y, z, w

  !$omp parallel default(private)
    !$omp parallel default(firstprivate)
      x = y
      w = w + 1
    !$omp end parallel

    !$omp parallel default(private) shared(z)
      w = x + z
    !$omp end parallel
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPnested_default_clause_test3
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_W:.*]], {{.*}} {{.*}}#0 -> %[[PRIVATE_Z:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test3Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFnested_default_clause_test3Ew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_test3Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_PRIVATE_X:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_PRIVATE_Y:.*]] : {{.*}}) {
!CHECK: %[[INNER_PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_X]] {uniq_name = "_QFnested_default_clause_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test3Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[INNER_PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[INNER_PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel {
!CHECK: %[[TEMP_1:.*]] = fir.load %[[PRIVATE_X_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[TEMP_2:.*]] = fir.load %[[PRIVATE_Z_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[TEMP_3:.*]] = arith.addi %[[TEMP_1]], %[[TEMP_2]] : i32
!CHECK: hlfir.assign %[[TEMP_3]] to %[[PRIVATE_W_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: }
subroutine nested_default_clause_test3
  integer :: x, y, z, w

  !$omp parallel default(private)
    !$omp parallel default(firstprivate)
      x = y
    !$omp end parallel

    !$omp parallel default(shared)
      w = x + z
    !$omp end parallel
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPnested_default_clause_test4
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFnested_default_clause_test4Ex"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFnested_default_clause_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFnested_default_clause_test4Ey"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFnested_default_clause_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_X:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[PRIVATE_Y:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.single {
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
subroutine nested_default_clause_test4
  integer :: x, y

  !$omp parallel default(firstprivate)
    !$omp single
      x = y
    !$omp end single
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPnested_default_clause_test5
!CHECK: omp.parallel {

!CHECK: %[[CONST_LB:.*]] = arith.constant 1 : i32
!CHECK: %[[CONST_UB:.*]] = arith.constant 50 : i32
!CHECK: %[[CONST_STEP:.*]] = arith.constant 1 : i32
! CHECK: omp.wsloop private(@{{.*}} %{{.*}} -> %[[X_ALLOCA:.*]], @{{.*}} %{{.*}} -> %[[LOOP_VAR_ALLOCA:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK: omp.loop_nest (%[[ARG:.*]]) : i32 = (%[[CONST_LB]]) to (%[[CONST_UB]]) inclusive step (%[[CONST_STEP]]) {
!CHECK: %[[X_DECLARE:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {{.*}}
!CHECK: %[[LOOP_VAR_DECLARE:.*]]:2 = hlfir.declare %[[LOOP_VAR_ALLOCA]] {{.*}}
!CHECK: hlfir.assign %[[ARG]] to %[[LOOP_VAR_DECLARE]]#0 : i32, !fir.ref<i32>
!CHECK: %[[LOADED_X:.*]] = fir.load %[[X_DECLARE]]#0 : !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 1 : i32
!CHECK: %[[RESULT:.*]] = arith.addi %[[LOADED_X]], %[[CONST]] : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[X_DECLARE]]#0 : i32, !fir.ref<i32>
!CHECK: omp.yield
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
subroutine nested_default_clause_test5
  integer :: i, x

  !$omp parallel do private(x)
    do i=1, 50
      x = x + 1
    end do
  !$omp end parallel do
end subroutine

!CHECK-LABEL: func @_QPnested_default_clause_test6
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[X_VAR:.*]], {{.*}} {{.*}}#0 -> %[[Y_VAR:.*]], {{.*}} {{.*}}#0 -> %[[Z_VAR:.*]] : {{.*}}) {
!CHECK: %[[X_VAR_DECLARE:.*]]:2 = hlfir.declare %[[X_VAR]] {{.*}}

!CHECK: %[[Y_VAR_DECLARE:.*]]:2 = hlfir.declare %[[Y_VAR]] {{.*}}

!CHECK: %[[Z_VAR_DECLARE:.*]]:2 = hlfir.declare %[[Z_VAR]] {{.*}}

!CHECK: %[[CONST_LB:.*]] = arith.constant 1 : i32
!CHECK: %[[CONST_UB:.*]] = arith.constant 10 : i32
!CHECK: %[[CONST_STEP:.*]] = arith.constant 1 : i32
! CHECK: omp.wsloop private(@{{.*}} %{{.*}} -> %[[LOOP_VAR:.*]] : !fir.ref<i32>) {
!CHECK: omp.loop_nest (%[[ARG:.*]]) : i32 = (%[[CONST_LB]]) to (%[[CONST_UB]]) inclusive step (%[[CONST_STEP]]) {
!CHECK: %[[LOOP_VAR_DECLARE:.*]]:2 = hlfir.declare %[[LOOP_VAR]] {{.*}}
!CHECK: hlfir.assign %[[ARG]] to %[[LOOP_VAR_DECLARE]]#0 : i32, !fir.ref<i32>
!CHECK: %[[LOADED_X:.*]] = fir.load %[[X_VAR_DECLARE]]#0 : !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 1 : i32
!CHECK: %[[ADD:.*]] = arith.addi %[[LOADED_X]], %[[CONST]] : i32
!CHECK: hlfir.assign %[[ADD]] to %[[X_VAR_DECLARE]]#0 : i32, !fir.ref<i32>
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_Y_ALLOCA:.*]], {{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_Z_ALLOCA:.*]] : {{.*}}) {
!CHECK: %[[INNER_Y_DECLARE:.*]]:2 = hlfir.declare %[[INNER_Y_ALLOCA]] {{.}}
!CHECK: %[[INNER_Z_DECLARE:.*]]:2 = hlfir.declare %[[INNER_Z_ALLOCA]] {{.}}
!CHECK: %[[LOADED_Y:.*]] = fir.load %[[INNER_Y_DECLARE]]#0 : !fir.ref<i32>
!CHECK: %[[LOADED_Z:.*]] = fir.load %[[INNER_Z_DECLARE]]#0 : !fir.ref<i32>
!CHECK: %[[RESULT:.*]] = arith.addi %[[LOADED_Y]], %[[LOADED_Z]] : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[INNER_Y_DECLARE]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }
subroutine nested_default_clause_test6
  integer :: i, x, y, z

  !$omp parallel do default(private)
    do i = 1, 10
      x  = x + 1
      !$omp parallel default(firstprivate)
         y = y + z
      !$omp end parallel
    end do
  !$omp end parallel do
end subroutine

!CHECK: func.func @_QPskipped_default_clause_checks() {
!CHECK: %[[TYPE_ADDR:.*]] = fir.address_of(@_QFskipped_default_clause_checksE.n.i1) : !fir.ref<!fir.char<1,2>>
!CHECK: %[[VAL_CONST_2:.*]] = arith.constant 2 : index
!CHECK: %[[VAL_I1_DECLARE:.*]]:2 = hlfir.declare %[[TYPE_ADDR]] typeparams %[[VAL_CONST_2]] {{.*}}
!CHECK: %[[TYPE_ADDR_IT:.*]] = fir.address_of(@_QFskipped_default_clause_checksE.n.it) : !fir.ref<!fir.char<1,2>>
!CHECK: %[[VAL_CONST_2_0:.*]] = arith.constant 2 : index
!CHECK: %[[VAL_IT_DECLARE:.*]]:2 = hlfir.declare %[[TYPE_ADDR_IT]] typeparams %[[VAL_CONST_2_0]] {{.*}}
!CHECK: %[[VAL_I_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFskipped_default_clause_checksEi"}
!CHECK: %[[VAL_I_DECLARE:.*]]:2 = hlfir.declare %[[VAL_I_ALLOCA]] {{.*}}
!CHECK: %[[VAL_III_ALLOCA:.*]] = fir.alloca !fir.type<_QFskipped_default_clause_checksTit{i1:i32}> {bindc_name = "iii", uniq_name = "_QFskipped_default_clause_checksEiii"}
!CHECK: %[[VAL_III_DECLARE:.*]]:2 = hlfir.declare %[[VAL_III_ALLOCA]] {{.*}}
!CHECK: %[[VAL_X_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFskipped_default_clause_checksEx"}
!CHECK: %[[VAL_X_DECLARE:.*]]:2 = hlfir.declare %[[VAL_X_ALLOCA]] {{.*}}
!CHECK: %[[VAL_Y_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFskipped_default_clause_checksEy"}
!CHECK: %[[VAL_Y_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Y_ALLOCA]] {{.*}}
!CHECK: %[[VAL_Z_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFskipped_default_clause_checksEz"}
!CHECK: %[[VAL_Z_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Z_ALLOCA]] {{.*}}
subroutine skipped_default_clause_checks()
       integer :: x,y,z
       type it
         integer::i1
       end type
       type(it)::iii

!CHECK: omp.parallel {{.*}} {
!CHECK: omp.wsloop private({{.*}}) reduction(@min_i32 %[[VAL_Z_DECLARE]]#0 -> %[[PRV:.+]] : !fir.ref<i32>) {
!CHECK-NEXT: omp.loop_nest (%[[ARG:.*]]) {{.*}} {
!CHECK: omp.yield
!CHECK: }
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
       !$omp parallel do default(private) REDUCTION(MIN:z)
         do i = 1, 10
           x = x + MIN(y,x)
         enddo
       !$omp end parallel do

!CHECK: omp.parallel {
!CHECK: omp.terminator
!CHECK: }
       namelist /nam/i
       !$omp parallel default(private)
          write(1,nam )
       !$omp endparallel

!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[PRIVATE_III:.*]] : {{.*}}) {
!CHECK: %[[PRIVATE_III_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_III]] {{.*}}
!CHECK: %[[PRIVATE_ADDR:.*]] = fir.address_of(@_QQro._QFskipped_default_clause_checksTit.0) : !fir.ref<!fir.type<_QFskipped_default_clause_checksTit{i1:i32}>>
!CHECK: %[[PRIVATE_PARAM:.*]]:2 = hlfir.declare %[[PRIVATE_ADDR]] {{.*}}
!CHECK: hlfir.assign %[[PRIVATE_PARAM]]#0 to %[[PRIVATE_III_DECLARE]]#0 {{.*}}
!CHECK: omp.terminator
!CHECK: }
       !$omp parallel default(private)
          iii=it(11)
       !$omp end parallel
end subroutine

!CHECK: func.func @_QPthreadprivate_with_default() {
!CHECK: %[[VAR_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFthreadprivate_with_defaultEi"}
!CHECK: %[[VAR_I_DECLARE:.*]] = hlfir.declare %[[VAR_I]] {uniq_name = "_QFthreadprivate_with_defaultEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[BLK_ADDR:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<4xi8>>
!CHECK: %[[BLK_THREADPRIVATE_OUTER:.*]] = omp.threadprivate %[[BLK_ADDR]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK: %[[VAR_C:.*]] = arith.constant 0 : index
!CHECK: %[[BLK_REF:.*]] = fir.coordinate_of %[[BLK_THREADPRIVATE_OUTER]], %[[VAR_C]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[CONVERT:.*]] = fir.convert %[[BLK_REF]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[VAR_X_DECLARE:.*]] = hlfir.declare %[[CONVERT]] storage(%[[BLK_THREADPRIVATE_OUTER]][0]) {uniq_name = "_QFthreadprivate_with_defaultEx"} : (!fir.ref<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK:   %[[BLK_THREADPRIVATE_INNER:.*]] = omp.threadprivate %[[BLK_ADDR]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
!CHECK:   %[[VAR_C_INNER:.*]] = arith.constant 0 : index
!CHECK:   %[[BLK_REF_INNER:.*]] = fir.coordinate_of %[[BLK_THREADPRIVATE_INNER]], %[[VAR_C_INNER]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
!CHECK:   %[[CONVERT_INNER:.*]] = fir.convert %[[BLK_REF_INNER]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:   %[[VAR_X_DECLARE_INNER:.*]] = hlfir.declare %[[CONVERT_INNER]] storage(%[[BLK_THREADPRIVATE_INNER]][0]) {uniq_name = "_QFthreadprivate_with_defaultEx"} : (!fir.ref<i32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine threadprivate_with_default
        integer :: x
        common /blk/ x 
        !$omp threadprivate (/blk/)

        !$omp parallel do default(private)
           do i = 1, 4
              x = 4
           end do
        !$omp end parallel do
end subroutine

subroutine nested_constructs
!CHECK: %[[I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFnested_constructsEi"}
!CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]] {{.*}}
!CHECK: %[[J:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_constructsEj"}
!CHECK: %[[J_DECL:.*]]:2 = hlfir.declare %[[J]] {{.*}}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFnested_constructsEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {{.*}}
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFnested_constructsEz"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {{.*}}

    integer :: y, z
!CHECK: omp.parallel private({{.*firstprivate.*}} {{.*}}#0 -> %[[INNER_Y:.*]], {{.*}} {{.*}}#0 -> %[[INNER_Z:.*]], {{.*}} {{.*}}#0 -> %[[INNER_I:.*]], {{.*}} {{.*}}#0 -> %[[INNER_J:.*]] : {{.*}}) {

!CHECK: %[[INNER_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_Y]] {{.*}}

!CHECK: %[[INNER_Z_DECL:.*]]:2 = hlfir.declare %[[INNER_Z]] {{.*}}

!CHECK: %[[INNER_I_DECL:.*]]:2 = hlfir.declare %[[INNER_I]] {{.*}}

!CHECK: %[[INNER_J_DECL:.*]]:2 = hlfir.declare %[[INNER_J]] {{.*}}
    !$omp parallel default(private) firstprivate(y)
!CHECK: {{.*}} = fir.do_loop {{.*}} {
      do i = 1, 10
!CHECK: %[[CONST_1:.*]] = arith.constant 1 : i32
!CHECK: hlfir.assign %[[CONST_1]] to %[[INNER_Y_DECL]]#0 : i32, !fir.ref<i32>
        y = 1
!CHECK: {{.*}} = fir.do_loop {{.*}} {
        do j = 1, 10
!CHECK: %[[CONST_20:.*]] = arith.constant 20 : i32
!CHECK: hlfir.assign %[[CONST_20]] to %[[INNER_Z_DECL]]#0 : i32, !fir.ref<i32>
          z = 20
!CHECK: omp.parallel private({{.*}} {{.*}}#0 -> %[[NESTED_Y:.*]] : {{.*}}) {
!CHECK: %[[NESTED_Y_DECL:.*]]:2 = hlfir.declare %[[NESTED_Y]] {{.*}}
!CHECK: %[[CONST_2:.*]] = arith.constant 2 : i32
!CHECK: hlfir.assign %[[CONST_2]] to %[[NESTED_Y_DECL]]#0 : i32, !fir.ref<i32>
          !$omp parallel default(private)
             y = 2
          !$omp end parallel
        end do
      end do
    !$omp end parallel
end subroutine
