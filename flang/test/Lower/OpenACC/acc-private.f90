! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_lb4.ub9_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.box<!fir.array<?xi32>>):
! CHECK: } copy {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>):
! HLFIR:   %[[LB:.*]] = arith.constant 4 : index
! HLFIR:   %[[UB:.*]] = arith.constant 9 : index
! HLFIR:   %[[STEP:.*]] = arith.constant 1 : index
! HLFIR:   %[[C1:.*]] = arith.constant 1 : index
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[EXT0:.*]] = arith.subi %[[UB]], %[[LB]] : index
! HLFIR:   %[[EXT1:.*]] = arith.addi %[[EXT0]], %[[C1]] : index
! HLFIR:   %[[EXT2:.*]] = arith.divsi %[[EXT1]], %[[STEP]] : index
! HLFIR:   %[[CMP:.*]] = arith.cmpi sgt, %[[EXT2]], %[[C0]] : index
! HLFIR:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[EXT2]], %[[C0]] : index
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[SELECT]] : (index) -> !fir.shape<1>
! HLFIR:   %[[LEFT:.*]] = hlfir.designate %[[ARG0]] shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! HLFIR:   %[[RIGHT:.*]] = hlfir.designate %[[ARG1]] shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! HLFIR:   hlfir.assign %[[LEFT]] to %[[RIGHT]] : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
! HLFIR:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! HLFIR:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1 {bindc_name = ".tmp", uniq_name = ""}
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! HLFIR:   acc.yield %[[DECL]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! HLFIR:   %[[DES_V1:.*]] = hlfir.designate %[[ARG0]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! HLFIR:   %[[DES_V2:.*]] = hlfir.designate %[[ARG1]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! HLFIR:   hlfir.assign %[[DES_V1]] to %[[DES_V2]] : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
! HLFIR:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_UxUx2xi32 : !fir.ref<!fir.array<?x?x2xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<?x?x2xi32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[ARG1]], %[[ARG2]], %[[ARG3]] : (index, index, index) -> !fir.shape<3>
! HLFIR:   %[[ALLOCA:.*]] = fir.alloca !fir.array<?x?x2xi32>, %[[ARG1]], %[[ARG2]], %[[ARG3]]
! HLFIR:   %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<?x?x2xi32>>, !fir.shape<3>) -> (!fir.box<!fir.array<?x?x2xi32>>, !fir.ref<!fir.array<?x?x2xi32>>)
! HLFIR:   acc.yield %[[DECL]]#0 : !fir.box<!fir.array<?x?x2xi32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_ptr_Uxi32 : !fir.box<!fir.ptr<!fir.array<?xi32>>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.ptr<!fir.array<?xi32>>>):
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! HLFIR:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_heap_Uxi32 : !fir.box<!fir.heap<!fir.array<?xi32>>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.heap<!fir.array<?xi32>>>):
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! HLFIR:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1 {bindc_name = ".tmp", uniq_name = ""}
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! HLFIR:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! HLFIR:   acc.yield %[[DECLARE:.*]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_lb50.ub99_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
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

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_ext100_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
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
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<50xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
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
! CHECK: acc.parallel firstprivate(@firstprivatization_section_ext100_ref_100xf32 -> %[[FP_B]] : !fir.ref<!fir.array<100xf32>>)
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
! CHECK: acc.parallel firstprivate(@firstprivatization_section_lb50.ub99_ref_50xf32 -> %[[FP_B]] : !fir.ref<!fir.array<50xf32>>)

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
! HLFIR: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFacc_private_assumed_shapeEa"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! HLFIR: %[[ADDR:.*]] = fir.box_addr %[[DECL_A]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! HLFIR: %[[PRIVATE:.*]] = acc.private varPtr(%[[ADDR]] : !fir.ref<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "a"}
! HLFIR: acc.parallel private(@privatization_box_Uxi32 -> %[[PRIVATE]] : !fir.ref<!fir.array<?xi32>>) {

subroutine acc_private_allocatable_array(a, n)
  integer, allocatable :: a(:)
  integer :: i, n

  !$acc parallel loop private(a)
  do i = 1, n
    a(i) = i
  end do

  !$acc serial private(a)
  a(i) = 1
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_allocatable_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}
! HLFIR: %[[DECLA_A:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFacc_private_allocatable_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! HLFIR: %[[BOX:.*]] = fir.load %[[DECLA_A]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR: %[[PRIVATE:.*]] = acc.private varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.heap<!fir.array<?xi32>> {name = "a"}
! HLFIR: acc.parallel private(@privatization_box_heap_Uxi32 -> %[[PRIVATE]] : !fir.heap<!fir.array<?xi32>>)
! HLFIR: acc.serial private(@privatization_box_heap_Uxi32 -> %{{.*}} : !fir.heap<!fir.array<?xi32>>)

subroutine acc_private_pointer_array(a, n)
  integer, pointer :: a(:)
  integer :: i, n

  !$acc parallel loop private(a)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_pointer_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
! HLFIR: %[[DECL_A:.*]]:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFacc_private_pointer_arrayEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! HLFIR: %[[BOX:.*]] = fir.load %[[DECLA_A]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! HLFIR: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! HLFIR: %[[PRIVATE:.*]] = acc.private varPtr(%[[BOX_ADDR]] : !fir.ptr<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ptr<!fir.array<?xi32>> {name = "a"}
! HLFIR: acc.parallel private(@privatization_box_ptr_Uxi32 -> %[[PRIVATE]] : !fir.ptr<!fir.array<?xi32>>)

subroutine acc_private_dynamic_extent(a, n)
  integer :: n, i
  integer :: a(n, n, 2)

  !$acc parallel loop private(a)
  do i = 1, n
    a(i, i, 1) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_dynamic_extent(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?x?x2xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! HLFIR: %[[DECL_N:.*]]:2 = hlfir.declare %arg1 {uniq_name = "_QFacc_private_dynamic_extentEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! HLFIR: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]](%16) {uniq_name = "_QFacc_private_dynamic_extentEa"} : (!fir.ref<!fir.array<?x?x2xi32>>, !fir.shape<3>) -> (!fir.box<!fir.array<?x?x2xi32>>, !fir.ref<!fir.array<?x?x2xi32>>)
! HLFIR: %[[PRIV:.*]] = acc.private varPtr(%[[DECL_A]]#1 : !fir.ref<!fir.array<?x?x2xi32>>) bounds(%{{.*}}, %{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<?x?x2xi32>> {name = "a"}
! HLFIR: acc.parallel private(@privatization_ref_UxUx2xi32 -> %[[PRIV]] : !fir.ref<!fir.array<?x?x2xi32>>)

subroutine acc_firstprivate_assumed_shape(a, n)
  integer :: a(:), i, n

  !$acc parallel loop firstprivate(a)
  do i = 1, n
    a(i) = i
  end do
end subroutine

subroutine acc_firstprivate_assumed_shape_with_section(a, n)
  integer :: a(:), i, n

  !$acc parallel loop firstprivate(a(5:10))
  do i = 1, n
    a(i) = i
  end do
end subroutine
