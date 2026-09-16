! This test checks the lowering of parallel do

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - \
! RUN: | FileCheck %s

! The string "EXPECTED" denotes the expected FIR

! CHECK: omp.parallel  private(@{{.*}} %{{.*}} -> %[[PRIVATE_Y:.*]], @{{.*}} %{{.*}} -> %[[PRIVATE_Z:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: %[[const_1:.*]] = arith.constant 1 : i32
! CHECK: %[[const_2:.*]] = arith.constant 10 : i32
! CHECK: %[[const_3:.*]] = arith.constant 1 : i32
! CHECK: omp.wsloop private(@{{.*}} %{{.*}} -> %[[TEMP:.*]] : !fir.ref<i32>) {
! CHECK-NEXT: omp.loop_nest (%[[ARG:.*]]) : i32 = (%[[const_1]]) to (%[[const_2]]) inclusive step (%[[const_3]]) {
! CHECK: %[[TEMP_DECL:.*]]:2 = hlfir.declare %[[TEMP]]
! CHECK: hlfir.assign %[[ARG]] to %[[TEMP_DECL]]#0 : i32, !fir.ref<i32>
! EXPECTED: %[[temp_1:.*]] = fir.load %[[PRIVATE_Z]] : !fir.ref<i32>
! CHECK: %[[temp_1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %[[temp_2:.*]] = fir.load %[[TEMP_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[result:.*]] = arith.addi %[[temp_1]], %[[temp_2]] : i32
! EXPECTED: hlfir.assign %[[result]] to %[[PRIVATE_Y]] : i32, !fir.ref<i32>
! CHECK: hlfir.assign %[[result]] to %{{.*}} : i32, !fir.ref<i32>
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
