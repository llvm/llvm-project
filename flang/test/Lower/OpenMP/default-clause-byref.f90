! This test checks lowering of OpenMP parallel directive
! with `DEFAULT` clause present.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --force-byref-reduction %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-hlfir --force-byref-reduction %s -o - | FileCheck %s


!CHECK: func @_QQmain() attributes {fir.bindc_name = "default_clause_lowering"} {
!CHECK: %[[W:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFEw"}
!CHECK: %[[W_DECL:.*]]:2 = hlfir.declare %[[W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[CONST]] to %[[PRIVATE_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFEw"}
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

!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(none) private(x, y)
        x = y
    !$omp end parallel

!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(firstprivate) firstprivate(y)
        x = y
    !$omp end parallel

!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFEw"}
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[W_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_W_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
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
!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFEw"}
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[W_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_W_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
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

subroutine nested_default_clause_tests
    integer :: x, y, z, w, k, a
!CHECK: %[[K:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFnested_default_clause_testsEk"}
!CHECK: %[[K_DECL:.*]]:2 = hlfir.declare %[[K]] {uniq_name = "_QFnested_default_clause_testsEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[W:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[W_DECL:.*]]:2 = hlfir.declare %[[W]] {uniq_name = "_QFnested_default_clause_testsEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFnested_default_clause_testsEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: %[[PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_testsEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_K:.*]] = fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFnested_default_clause_testsEk"}
!CHECK: %[[PRIVATE_K_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_K]] {uniq_name = "_QFnested_default_clause_testsEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK: %[[INNER_PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[INNER_PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[INNER_PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST:.*]] = arith.constant 20 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_Y_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: %[[CONST:.*]] = arith.constant 10 : i32
!CHECK: hlfir.assign %[[CONST]] to %[[INNER_PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel   {
!CHECK: %[[INNER_PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[INNER_PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_W]] {uniq_name = "_QFnested_default_clause_testsEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INNER_PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: %[[INNER_PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_testsEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Z_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[INNER_PRIVATE_Z_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[INNER_PRIVATE_K:.*]] = fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFnested_default_clause_testsEk"}
!CHECK: %[[INNER_PRIVATE_K_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_K]] {uniq_name = "_QFnested_default_clause_testsEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_K_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[INNER_PRIVATE_K_DECL]]#0 temporary_lhs : i32, !fir.ref<i32> 
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
    
    
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: %[[PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_testsEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFnested_default_clause_testsEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_INNER_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_INNER_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_INNER_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_INNER_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[INNER_PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[INNER_PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[INNER_PRIVATE_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[TEMP:.*]] = fir.load %[[INNER_PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_INNER_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_INNER_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[PRIVATE_INNER_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_INNER_W]] {uniq_name = "_QFnested_default_clause_testsEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_INNER_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_INNER_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_INNER_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP_1:.*]] = fir.load %[[PRIVATE_INNER_X_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[TEMP_2:.*]] = fir.load %[[PRIVATE_Z_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[RESULT:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
!CHECK: hlfir.assign %[[RESULT]] to %[[PRIVATE_INNER_W_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
    !$omp parallel default(private)
        !$omp parallel default(firstprivate)
            x = y
        !$omp end parallel

        !$omp parallel default(private) shared(z)
            w = x + z
        !$omp end parallel
    !$omp end parallel    
    
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[PRIVATE_W_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_W]] {uniq_name = "_QFnested_default_clause_testsEw"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: %[[PRIVATE_Z_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Z]] {uniq_name = "_QFnested_default_clause_testsEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[INNER_PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[INNER_PRIVATE_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[INNER_PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[INNER_PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[INNER_PRIVATE_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
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
    !$omp parallel default(private)
		!$omp parallel default(firstprivate)
			x = y
		!$omp end parallel

		!$omp parallel default(shared)
			w = x + z
		!$omp end parallel
	!$omp end parallel

!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X]] {uniq_name = "_QFnested_default_clause_testsEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y]] {uniq_name = "_QFnested_default_clause_testsEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[TEMP:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: omp.single {
!CHECK: %[[TEMP:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[TEMP]] to %[[PRIVATE_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: } 
	!$omp parallel default(firstprivate)
		!$omp single
			x = y
		!$omp end single
	!$omp end parallel
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

!CHECK: omp.parallel {
!CHECK: omp.wsloop byref reduction(@min_i_32_byref %[[VAL_Z_DECLARE]]#0 -> %[[PRV:.+]] : !fir.ref<i32>) for (%[[ARG:.*]]) {{.*}} {
!CHECK: omp.yield
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

!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_III_ALLOCA:.*]] = fir.alloca !fir.type<_QFskipped_default_clause_checksTit{i1:i32}> {{.*}}
!CHECK: %[[PRIVATE_III_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_III_ALLOCA]] {{.*}}
!CHECK: %[[PRIVATE_ADDR:.*]] = fir.address_of(@_QQro._QFskipped_default_clause_checksTit.0) : !fir.ref<!fir.type<_QFskipped_default_clause_checksTit{i1:i32}>>
!CHECK: %[[PRIVATE_PARAM:.*]]:2 = hlfir.declare %[[PRIVATE_ADDR]] {{.*}}
!CHECK: hlfir.assign %[[PRIVATE_PARAM]]#0 to %[[PRIVATE_III_DECLARE]]#0 {{.*}}
!CHECK: omp.terminator
!CHECK: }
       !$omp parallel default(private)
          iii=it(11)
       !$omp end parallel
end subroutine
