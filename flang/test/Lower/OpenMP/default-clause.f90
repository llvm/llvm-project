! This test checks lowering of OpenMP parallel directive
! with `DEFAULT` clause present.

! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s


!CHECK: func @_QQmain() {
!CHECK: %[[W:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFEw"}
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFEz"}
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFEw"}
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[const:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[const]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: omp.barrier
!CHECK: %[[const:.*]] = arith.constant 2 : i32
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.muli %[[const]], %[[temp]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_W]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 45 : i32
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[const]] : i32
!CHECK: fir.store %[[result]] to %[[Z]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

program default_clause_lowering
    integer :: x, y, z, w

    !$omp parallel default(private) firstprivate(x) shared(z)
        x = y * 2
        z = w + 45
    !$omp end parallel

!CHECK: omp.parallel {
!CHECK: %[[temp:.*]] = fir.load %[[Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(shared)
        x = y
    !$omp end parallel

!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(none) private(x, y)
        x = y
    !$omp end parallel

!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[temp:.*]] = fir.load %[[Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(firstprivate) firstprivate(y)
        x = y
    !$omp end parallel

!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFEw"}
!CHECK: %[[temp:.*]] = fir.load %[[W]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_W]] : !fir.ref<i32>
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[temp:.*]] = fir.load %[[Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[const:.*]] = arith.constant 2 : i32
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.muli %[[const]], %[[temp]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_W]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 45 : i32
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[const]] : i32
!CHECK: fir.store %[[result]] to %[[Z]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }

    !$omp parallel default(firstprivate) private(x) shared(z)
        x = y * 2
        z = w + 45
    !$omp end parallel

!CHECK: omp.parallel   {
!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFEw"}
!CHECK: %[[temp:.*]] = fir.load %[[W]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_W]] : !fir.ref<i32>
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_W]] : !fir.ref<i32>
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

!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFEz"}
!CHECK: %[[TEMP:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK: %[[const_1:.*]] = arith.constant 1 : i32
!CHECK: %[[const_2:.*]] = arith.constant 10 : i32
!CHECK: %[[const_3:.*]] = arith.constant 1 : i32
!CHECK: omp.wsloop   for  (%[[ARG:.*]]) : i32 = (%[[const_1]]) to (%[[const_2]]) inclusive step (%[[const_3]]) {
!CHECK: fir.store %[[ARG]] to %[[TEMP]] : !fir.ref<i32>
!CHECK: %[[temp_1:.*]] = fir.load %[[PRIVATE_Z]] : !fir.ref<i32>
!CHECK: %[[temp_2:.*]] = fir.load %[[TEMP]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.addi %[[temp_1]], %[[temp_2]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: omp.yield
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
    !$omp parallel do default(private)
        do x = 1, 10
            y = z + x
        enddo
    !$omp end parallel do
end program default_clause_lowering

subroutine nested_default_clause_tests
    integer :: x, y, z, w, k, a
    
!CHECK: %[[A:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFnested_default_clause_testsEa"}
!CHECK: %[[K:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFnested_default_clause_testsEk"}
!CHECK: %[[W:.*]] = fir.alloca i32 {bindc_name = "w", uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: omp.parallel   {
!CHECK: %[[PRIVATE_A:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFnested_default_clause_testsEa"}
!CHECK: %[[PRIVATE_K:.*]] = fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFnested_default_clause_testsEk"}
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: omp.barrier
!CHECK: omp.parallel {
!CHECK: %[[INNER_PRIVATE_A:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFnested_default_clause_testsEa"}
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[INNER_PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[const:.*]] = arith.constant 20 : i32
!CHECK: fir.store %[[const]] to %[[INNER_PRIVATE_Y]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 10 : i32
!CHECK: fir.store %[[const]] to %[[INNER_PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.parallel {
!CHECK: %[[const:.*]] = arith.constant 10 : i32
!CHECK: fir.store %[[const]] to %[[INNER_PRIVATE_A]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel   {
!CHECK: %[[INNER_PRIVATE_K:.*]] = fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFnested_default_clause_testsEk"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_K]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[INNER_PRIVATE_K]] : !fir.ref<i32>
!CHECK: %[[INNER_PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[INNER_PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Z]]
!CHECK: fir.store %[[temp]] to %[[INNER_PRIVATE_Z]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[const:.*]] = arith.constant 30 : i32
!CHECK: fir.store %[[const]] to %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 40 : i32
!CHECK: fir.store %[[const]] to %[[INNER_PRIVATE_W]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 50 : i32
!CHECK: fir.store %[[const]] to %[[INNER_PRIVATE_Z]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 40 : i32
!CHECK: fir.store %[[const]] to %[[INNER_PRIVATE_K]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
    !$omp parallel  firstprivate(x) private(y) shared(w) default(private)  
        !$omp parallel default(private)
           y = 20
           x = 10
           !$omp parallel
                a = 10
           !$omp end parallel
        !$omp end parallel 

        !$omp parallel default(firstprivate) shared(y) private(w) 
            y = 30
            w = 40 
            z = 50
            k = 40
        !$omp end parallel
    !$omp end parallel
    
    
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"} 
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_INNER_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_INNER_X]] : !fir.ref<i32>
!CHECK: %[[INNER_PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[INNER_PRIVATE_Y]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[temp:.*]] = fir.load %[[INNER_PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_INNER_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel {
!CHECK: %[[PRIVATE_INNER_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"} 
!CHECK: %[[PRIVATE_INNER_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[temp_1:.*]] = fir.load %[[PRIVATE_INNER_X]] : !fir.ref<i32>
!CHECK: %[[temp_2:.*]] = fir.load %[[PRIVATE_Z]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_INNER_W]] : !fir.ref<i32>
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
!CHECK: %[[PRIVATE_W:.*]] = fir.alloca i32 {bindc_name = "w", pinned, uniq_name = "_QFnested_default_clause_testsEw"}
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFnested_default_clause_testsEz"}
!CHECK: omp.parallel {
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFnested_default_clause_testsEx"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[INNER_PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[INNER_PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[INNER_PRIVATE_Y]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[temp:.*]] = fir.load %[[INNER_PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[INNER_PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.parallel {
!CHECK: %[[temp_1:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[temp_2:.*]] = fir.load %[[PRIVATE_Z]] : !fir.ref<i32>
!CHECK: %[[temp_3:.*]] = arith.addi %[[temp_1]], %[[temp_2]] : i32
!CHECK: fir.store %[[temp_3]] to %[[PRIVATE_W]] : !fir.ref<i32>
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
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFnested_default_clause_testsEy"}
!CHECK: %[[temp:.*]] = fir.load %[[Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: omp.single {
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_Y]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: } 
	!$omp parallel default(firstprivate)
		!$omp single
			x = y
		!$omp end single
	!$omp end parallel
end subroutine
