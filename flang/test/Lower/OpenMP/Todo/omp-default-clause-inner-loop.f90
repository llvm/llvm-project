! This test checks the lowering of parallel do

! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: bbc -fopenmp -emit-fir -hlfir=false %s -o - | FileCheck %s

! The string "EXPECTED" denotes the expected FIR

! CHECK: omp.parallel   {
! EXPECTED: %[[PRIVATE_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFEy"}
! EXPECTED: %[[PRIVATE_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFEz"}
! CHECK: %[[TEMP:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK: %[[const_1:.*]] = arith.constant 1 : i32
! CHECK: %[[const_2:.*]] = arith.constant 10 : i32
! CHECK: %[[const_3:.*]] = arith.constant 1 : i32
! CHECK: omp.wsloop   for  (%[[ARG:.*]]) : i32 = (%[[const_1]]) to (%[[const_2]]) inclusive step (%[[const_3]]) {
! CHECK: fir.store %[[ARG]] to %[[TEMP]] : !fir.ref<i32>
! EXPECTED: %[[temp_1:.*]] = fir.load %[[PRIVATE_Z]] : !fir.ref<i32>
! CHECK: %[[temp_1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %[[temp_2:.*]] = fir.load %[[TEMP]] : !fir.ref<i32>
! CHECK: %[[result:.*]] = arith.addi %[[temp_1]], %[[temp_2]] : i32
! EXPECTED: fir.store %[[result]] to %[[PRIVATE_Y]] : !fir.ref<i32>
! CHECK: fir.store %[[result]] to %{{.*}} : !fir.ref<i32>
! CHECK: omp.yield
! CHECK: }
! CHECK: omp.terminator
! CHECK: }
subroutine nested_default_clause()
	integer x, y, z
	!$omp parallel do default(private)
		do x = 1, 10
			y = z + x
		enddo
	!$omp end parallel do
end subroutine
