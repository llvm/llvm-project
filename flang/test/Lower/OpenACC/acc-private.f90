! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<50xf32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.array<50xf32>>, %[[DST:.*]]: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 49 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[COORD0:.*]] = fir.coordinate_of %[[SRC]], %[[IV0]] : (!fir.ref<!fir.array<50xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[COORD1:.*]] = fir.coordinate_of %[[DST]], %[[IV0]] : (!fir.ref<!fir.array<50xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[VALUE:.*]] = fir.load %[[COORD0]] : !fir.ref<f32>
! CHECK:     fir.store %[[VALUE]] to %[[COORD1]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xf32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.array<100xf32>>, %[[DST:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP1:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %c0 to %c99 step %c1 {
! CHECK:     %[[COORD0:.*]] = fir.coordinate_of %[[SRC]], %[[IV0]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[COORD1:.*]] = fir.coordinate_of %[[DST]], %[[IV0]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[VALUE:.*]] = fir.load %[[COORD0]] : !fir.ref<f32>
! CHECK:     fir.store %[[VALUE]] to %[[COORD1]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_i32 : !fir.ref<i32> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK: %[[ALLOCA:.*]] = fir.alloca i32
! CHECK: acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<i32>, %[[DST:.*]]: !fir.ref<i32>):
! CHECK:   %[[VALUE:.*]] = fir.load %[[SRC]] : !fir.ref<i32>
! CHECK:   fir.store %[[VALUE]] to %[[DST]] : !fir.ref<i32>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! CHECK: acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<50xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: }

program acc_private
  integer :: i, c
  integer, parameter :: n = 100
  real, dimension(n) :: a, b

! CHECK: %[[B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<100xf32>>
! CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}

  !$acc loop private(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C_PRIVATE:.*]] = acc.private varPtr(%[[C]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "c"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[C_PRIVATE]] : !fir.ref<i32>)
! CHECK: acc.yield

  !$acc loop private(b)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %{{.*}}, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK: acc.loop private(@privatization_ref_100xf32 -> %[[B_PRIVATE]] : !fir.ref<!fir.array<100xf32>>) {
! CHECK: acc.yield

  !$acc loop private(b(1:50))
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.constant 49 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(1:50)"}
! CHECK: acc.loop private(@privatization_ref_50xf32 -> %[[B_PRIVATE]] : !fir.ref<!fir.array<50xf32>>)

  !$acc parallel loop firstprivate(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO
  
! CHECK: %[[FP_C:.*]] = acc.firstprivate varPtr(%[[C]] : !fir.ref<i32>)   -> !fir.ref<i32> {name = "c"}
! CHECK: acc.parallel firstprivate(@firstprivatization_ref_i32 -> %[[FP_C]] : !fir.ref<i32>)
! CHECK: acc.yield

  !$acc parallel loop firstprivate(b)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %{{.*}}, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK: acc.parallel firstprivate(@firstprivatization_ref_100xf32 -> %[[FP_B]] : !fir.ref<!fir.array<100xf32>>)
! CHECK: acc.yield

  !$acc parallel loop firstprivate(b(51:100))
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 50 : index
! CHECK: %[[UB:.*]] = arith.constant 99 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(51:100)"}
! CHECK: acc.parallel firstprivate(@firstprivatization_ref_50xf32 -> %[[FP_B]] : !fir.ref<!fir.array<50xf32>>)

end program
