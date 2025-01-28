! This test checks the lowering of parallel do

! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - \
! RUN: | FileCheck %s

! RUN: bbc -fopenmp -emit-fir -hlfir=false %s -o - \
! RUN: | FileCheck %s

! The string "EXPECTED" denotes the expected FIR

! CHECK: omp.parallel  private(@{{.*}} %{{.*}} -> %[[PRIVATE_Y:.*]], @{{.*}} %{{.*}} -> %[[PRIVATE_Y:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: %[[TEMP:.*]] = fir.alloca i32 {bindc_name = "x", pinned, {{.*}}}
! CHECK: %[[const_1:.*]] = arith.constant 1 : i32
! CHECK: %[[const_2:.*]] = arith.constant 10 : i32
! CHECK: %[[const_3:.*]] = arith.constant 1 : i32
! CHECK: omp.wsloop {
! CHECK-NEXT: omp.loop_nest (%[[ARG:.*]]) : i32 = (%[[const_1]]) to (%[[const_2]]) inclusive step (%[[const_3]]) {
! CHECK: fir.store %[[ARG]] to %[[TEMP]] : !fir.ref<i32>
! EXPECTED: %[[temp_1:.*]] = fir.load %[[PRIVATE_Z]] : !fir.ref<i32>
! CHECK: %[[temp_1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %[[temp_2:.*]] = fir.load %[[TEMP]] : !fir.ref<i32>
! CHECK: %[[result:.*]] = arith.addi %[[temp_1]], %[[temp_2]] : i32
! EXPECTED: fir.store %[[result]] to %[[PRIVATE_Y]] : !fir.ref<i32>
! CHECK: fir.store %[[result]] to %{{.*}} : !fir.ref<i32>
! CHECK: omp.yield
! CHECK: }
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
