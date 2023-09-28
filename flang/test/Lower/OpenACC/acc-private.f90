! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

! CHECK-LABEL: acc.private.recipe @"privatization_box_?xi32" : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! HLFIR:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! HLFIR:   acc.yield %[[DECLARE:.*]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<50xf32>>
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
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xf32>>
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
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) 
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32> 
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<i32>, %[[DST:.*]]: !fir.ref<i32>):
! CHECK:   %[[VALUE:.*]] = fir.load %[[SRC]] : !fir.ref<i32>
! CHECK:   fir.store %[[VALUE]] to %[[DST]] : !fir.ref<i32>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<50xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) 
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32> 
! CHECK: }

program acc_private
  integer :: i, c
  integer, parameter :: n = 100
  real, dimension(n) :: a, b

! CHECK: %[[B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<100xf32>>
! HLFIR: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}
! HLFIR: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]

  !$acc loop private(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! FIR: %[[C_PRIVATE:.*]] = acc.private varPtr(%[[C]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "c"}
! HLFIR: %[[C_PRIVATE:.*]] = acc.private varPtr(%[[DECLC]]#1 : !fir.ref<i32>) -> !fir.ref<i32> {name = "c"}
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
! FIR: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! HLFIR: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
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
! FIR: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(1:50)"}
! HLFIR: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(1:50)"}
! CHECK: acc.loop private(@privatization_ref_50xf32 -> %[[B_PRIVATE]] : !fir.ref<!fir.array<50xf32>>)

  !$acc parallel loop firstprivate(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! FIR: %[[FP_C:.*]] = acc.firstprivate varPtr(%[[C]] : !fir.ref<i32>)   -> !fir.ref<i32> {name = "c"}
! HLFIR: %[[FP_C:.*]] = acc.firstprivate varPtr(%[[DECLC]]#1 : !fir.ref<i32>)   -> !fir.ref<i32> {name = "c"}
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
! FIR: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! HLFIR: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
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
! FIR: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(51:100)"}
! HLFIR: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(51:100)"}
! CHECK: acc.parallel firstprivate(@firstprivatization_ref_50xf32 -> %[[FP_B]] : !fir.ref<!fir.array<50xf32>>)

end program

subroutine acc_private_assumed_shape(a, n)
  integer :: a(:), i, n

  !$acc parallel loop private(a)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_assumed_shape(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
! HLFIR: %[[DECL_A:.*]]:2 = hlfir.declare %arg0 {uniq_name = "_QFacc_private_assumed_shapeEa"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! HLFIR: %[[ADDR:.*]] = fir.box_addr %[[DECL_A]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! HLFIR: %[[PRIVATE:.*]] = acc.private varPtr(%[[ADDR]] : !fir.ref<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "a"}
! HLFIR: acc.parallel private(@"privatization_box_?xi32" -> %[[PRIVATE]] : !fir.ref<!fir.array<?xi32>>) {
