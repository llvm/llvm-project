! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: acc.private.recipe @privatization_ref_10xf32 : !fir.ref<!fir.array<10xf32>> init {
! CHECK:   ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>):
! CHECK:   %[[C10:.*]] = arith.constant 10 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<10xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_box_UxUx2xi32 : !fir.box<!fir.array<?x?x2xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:   %[[DIM0:.*]]:3 = fir.box_dims %arg0, %c0{{.*}} : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:   %[[DIM1:.*]]:3 = fir.box_dims %arg0, %c1{{.*}} : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[DIM0]]#1, %[[DIM1]]#1, %c2{{.*}} : (index, index, index) -> !fir.shape<3>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?x?x2xi32>, %[[DIM0]]#1, %[[DIM1]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?x?x2xi32>>, !fir.shape<3>) -> (!fir.box<!fir.array<?x?x2xi32>>, !fir.heap<!fir.array<?x?x2xi32>>)
! CHECK:   acc.yield %[[DECL]]#0 : !fir.box<!fir.array<?x?x2xi32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?x?x2xi32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?x?x2xi32>>, %[[LB0:.*]]: index, %[[UB0:.*]]: index, %[[STEP0:.*]]: index, %[[LB1:.*]]: index, %[[UB1:.*]]: index, %[[STEP1:.*]]: index, %[[LB2:.*]]: index, %[[UB2:.*]]: index, %[[STEP2:.*]]: index):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.shape<3>
! CHECK:   %[[DES_SRC:.*]] = hlfir.designate %[[ARG0]] (%[[LB0]]:%[[UB0]]:%[[STEP0]], %[[LB1]]:%[[UB1]]:%[[STEP1]], %[[LB2]]:%[[UB2]]:%[[STEP2]]) shape %[[SHAPE]] : (!fir.box<!fir.array<?x?x2xi32>>, index, index, index, index, index, index, index, index, index, !fir.shape<3>) -> !fir.box<!fir.array<?x?x2xi32>>
! CHECK:   %[[DES_DST:.*]] = hlfir.designate %[[ARG1]] (%[[LB0]]:%[[UB0]]:%[[STEP0]], %[[LB1]]:%[[UB1]]:%[[STEP1]], %[[LB2]]:%[[UB2]]:%[[STEP2]]) shape %[[SHAPE]] : (!fir.box<!fir.array<?x?x2xi32>>, index, index, index, index, index, index, index, index, index, !fir.shape<3>) -> !fir.box<!fir.array<?x?x2xi32>>
! CHECK:   hlfir.assign %[[DES_SRC]] to %[[DES_DST]] : !fir.box<!fir.array<?x?x2xi32>>, !fir.box<!fir.array<?x?x2xi32>>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_lb4.ub9_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.box<!fir.array<?xi32>>):
! CHECK: } copy {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:   %[[LB:.*]] = arith.constant 4 : index
! CHECK:   %[[UB:.*]] = arith.constant 9 : index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   %[[C1:.*]] = arith.constant 1 : index
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[EXT0:.*]] = arith.subi %[[UB]], %[[LB]] : index
! CHECK:   %[[EXT1:.*]] = arith.addi %[[EXT0]], %[[C1]] : index
! CHECK:   %[[EXT2:.*]] = arith.divsi %[[EXT1]], %[[STEP]] : index
! CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %[[EXT2]], %[[C0]] : index
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[EXT2]], %[[C0]] : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[SELECT]] : (index) -> !fir.shape<1>
! CHECK:   %[[LEFT:.*]] = hlfir.designate %[[ARG0]] shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[RIGHT:.*]] = hlfir.designate %[[ARG1]] shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[LEFT]] to %[[RIGHT]] : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:   acc.yield %[[DECL]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[DES_V1:.*]] = hlfir.designate %[[ARG0]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[DES_V2:.*]] = hlfir.designate %[[ARG1]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   hlfir.assign %[[DES_V1]] to %[[DES_V2]] : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_UxUx2xi32 : !fir.box<!fir.array<?x?x2xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:   %[[DIM0:.*]]:3 = fir.box_dims %arg0, %c0{{.*}} : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:   %[[DIM1:.*]]:3 = fir.box_dims %arg0, %c1{{.*}} : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[DIM0]]#1, %[[DIM1]]#1, %c2{{.*}} : (index, index, index) -> !fir.shape<3>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?x?x2xi32>, %[[DIM0]]#1, %[[DIM1]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?x?x2xi32>>, !fir.shape<3>) -> (!fir.box<!fir.array<?x?x2xi32>>, !fir.heap<!fir.array<?x?x2xi32>>)
! CHECK:   acc.yield %[[DECL]]#0 : !fir.box<!fir.array<?x?x2xi32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_ptr_Uxi32 : !fir.box<!fir.ptr<!fir.array<?xi32>>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.ptr<!fir.array<?xi32>>>):
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_heap_Uxi32 : !fir.box<!fir.heap<!fir.array<?xi32>>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.heap<!fir.array<?xi32>>>):
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:   acc.yield %[[DECLARE:.*]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_lb50.ub99_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<50xf32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.array<50xf32>>, %[[DST:.*]]: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[DECL_SRC:.*]]:2 = hlfir.declare %[[SRC]](%[[SHAPE]]) {uniq_name = ""} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! CHECK:   %[[DECL_DST:.*]]:2 = hlfir.declare %[[DST]](%[[SHAPE]]) {uniq_name = ""} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! CHECK:   %[[DES_SRC:.*]] = hlfir.designate %[[DECL_SRC]]#0 shape %[[SHAPE:.*]] : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
! CHECK:   %[[DES_DST:.*]] = hlfir.designate %[[DECL_DST]]#0 shape %[[SHAPE:.*]] : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
! CHECK:   hlfir.assign %[[DES_SRC]] to %[[DES_DST]] : !fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_ext100_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xf32>>
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<!fir.array<100xf32>>, %[[DST:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[DECL_SRC:.*]]:2 = hlfir.declare %[[SRC]](%[[SHAPE]]) {uniq_name = ""} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:   %[[DECL_DST:.*]]:2 = hlfir.declare %[[DST]](%[[SHAPE]]) {uniq_name = ""} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:   %[[DES_SRC:.*]] = hlfir.designate %[[DECL_SRC]]#0 shape %[[SHAPE]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:   %[[DES_DST:.*]] = hlfir.designate %[[DECL_DST]]#0 shape %[[SHAPE]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:   hlfir.assign %[[DES_SRC]] to %[[DES_DST]] : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_i32 : !fir.ref<i32> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) 
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32> 
! CHECK: } copy {
! CHECK: ^bb0(%[[SRC:.*]]: !fir.ref<i32>, %[[DST:.*]]: !fir.ref<i32>):
! CHECK:   %[[VALUE:.*]] = fir.load %[[SRC]] : !fir.ref<i32>
! CHECK:   fir.store %[[VALUE]] to %[[DST]] : !fir.ref<i32>
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_50xf32 : !fir.ref<!fir.array<50xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<50xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<50xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<50xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<50xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>) 
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32> 
! CHECK: }

program acc_private
  integer :: i, c
  integer, parameter :: n = 100
  real, dimension(n) :: a, b

! CHECK: %[[B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<100xf32>>
! CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}
! CHECK: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]

  !$acc loop private(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C_PRIVATE:.*]] = acc.private varPtr(%[[DECLC]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {name = "c"}
! CHECK: acc.loop private({{.*}}@privatization_ref_i32 -> %[[C_PRIVATE]] : !fir.ref<i32>)
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
! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK: acc.loop private({{.*}}@privatization_ref_100xf32 -> %[[B_PRIVATE]] : !fir.ref<!fir.array<100xf32>>)
! CHECK: acc.yield

  !$acc loop private(b(1:50))
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.constant 49 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(1:50)"}
! CHECK: acc.loop private({{.*}}@privatization_ref_50xf32 -> %[[B_PRIVATE]] : !fir.ref<!fir.array<50xf32>>)

  !$acc parallel loop firstprivate(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[FP_C:.*]] = acc.firstprivate varPtr(%[[DECLC]]#0 : !fir.ref<i32>)   -> !fir.ref<i32> {name = "c"}
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
! CHECK: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
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
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<50xf32>> {name = "b(51:100)"}
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
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFacc_private_assumed_shapeEa"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: acc.parallel {
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[DECL_A]]#0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[ADDR]] : !fir.ref<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.loop private({{.*}}@privatization_box_Uxi32 -> %[[PRIVATE]] : !fir.ref<!fir.array<?xi32>>)

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
! CHECK: %[[DECLA_A:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFacc_private_allocatable_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK: acc.parallel {
! CHECK: %[[BOX:.*]] = fir.load %[[DECLA_A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.heap<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.loop private({{.*}}@privatization_box_heap_Uxi32 -> %[[PRIVATE]] : !fir.heap<!fir.array<?xi32>>)
! CHECK: acc.serial private(@privatization_box_heap_Uxi32 -> %{{.*}} : !fir.heap<!fir.array<?xi32>>)

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
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFacc_private_pointer_arrayEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK: acc.parallel {
! CHECK: %[[BOX:.*]] = fir.load %[[DECLA_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[BOX_ADDR]] : !fir.ptr<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ptr<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.loop private({{.*}}@privatization_box_ptr_Uxi32 -> %[[PRIVATE]] : !fir.ptr<!fir.array<?xi32>>)

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
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFacc_private_dynamic_extentEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {uniq_name = "_QFacc_private_dynamic_extentEa"} : (!fir.ref<!fir.array<?x?x2xi32>>, !fir.shape<3>) -> (!fir.box<!fir.array<?x?x2xi32>>, !fir.ref<!fir.array<?x?x2xi32>>)
! CHECK: acc.parallel {
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL_A]]#0 : (!fir.box<!fir.array<?x?x2xi32>>) -> !fir.ref<!fir.array<?x?x2xi32>>
! CHECK: %[[PRIV:.*]] = acc.private varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?x?x2xi32>>) bounds(%{{.*}}, %{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<?x?x2xi32>> {name = "a"}
! CHECK: acc.loop private({{.*}}@privatization_box_UxUx2xi32 -> %[[PRIV]] : !fir.ref<!fir.array<?x?x2xi32>>)

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

subroutine acc_firstprivate_dynamic_extent(a, n)
  integer :: n, i
  integer :: a(n, n, 2)

  !$acc parallel loop firstprivate(a)
  do i = 1, n
    a(i, i, 1) = i
  end do
end subroutine

! CHECK: acc.parallel firstprivate(@firstprivatization_box_UxUx2xi32 -> %{{.*}} : !fir.ref<!fir.array<?x?x2xi32>>)

module acc_declare_equivalent
  integer, parameter :: n = 10
  real :: v1(n)
  real :: v2(n)
  equivalence(v1(1), v2(1))
contains
  subroutine sub1()
    !$acc parallel private(v2)
    !$acc end parallel
  end subroutine
end module

! CHECK: acc.parallel private(@privatization_ref_10xf32 -> %{{.*}} : !fir.ref<!fir.array<10xf32>>)
