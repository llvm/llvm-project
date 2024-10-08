! This test checks lowering of OpenMP parallel DO, with the loop bound being
! a lastprivate variable

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK: func @_QPomp_do_lastprivate(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"})
subroutine omp_do_lastprivate(a)
  ! CHECK: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_do_lastprivateEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer::a
  integer::n
  n = a+1
  !$omp parallel do lastprivate(a)
  ! CHECK:  omp.parallel {

  ! CHECK: %[[A_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_do_lastprivateEa"}
  ! CHECK: %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFomp_do_lastprivateEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[I_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
  ! CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFomp_do_lastprivateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[LB:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[ARG1:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  ! CHECK-NEXT: fir.store %[[ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.call @_QPfoo(%[[I_PVT_DECL]]#1, %[[A_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK:      %[[NEXT_ARG1:.*]] = arith.addi %[[ARG1]], %[[STEP]] : i32
  ! CHECK:      %[[ZERO:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[STEP_DIR:.*]] = arith.cmpi slt, %[[STEP]], %[[ZERO]] : i32
  ! CHECK:      %[[LT_UB:.*]] = arith.cmpi slt, %[[NEXT_ARG1]], %[[UB]] : i32
  ! CHECK:      %[[GT_UB:.*]] = arith.cmpi sgt, %[[NEXT_ARG1]], %[[UB]] : i32
  ! CHECK:      %[[SEL:.*]] = arith.select %[[STEP_DIR]], %[[LT_UB]], %[[GT_UB]] : i1
  ! CHECK:      fir.if %[[SEL]] {
  ! CHECK:        fir.store %[[NEXT_ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:        %[[A_PVT_LOAD:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:        hlfir.assign %[[A_PVT_LOAD]] to %[[ARG0_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:      }

  ! CHECK-NEXT: omp.yield
  ! CHECK-NEXT: }
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: }
    do i=1, a
      call foo(i, a)
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG0_DECL]]#1) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(a)
end subroutine omp_do_lastprivate

! CHECK: func @_QPomp_do_lastprivate2(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"})
subroutine omp_do_lastprivate2(a, n)
  ! CHECK:  %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_do_lastprivate2Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:  %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_do_lastprivate2En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer::a
  integer::n
  n = a+1
  !$omp parallel do lastprivate(a, n)
  ! CHECK:  omp.parallel {

  ! CHECK: %[[A_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "a", pinned, {{.*}}}
  ! CHECK: %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate2Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[N_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "n", pinned, uniq_name = "_QFomp_do_lastprivate2En"}
  ! CHECK: %[[N_PVT_DECL:.*]]:2 = hlfir.declare %[[N_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[I_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
  ! CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[LB:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[UB:.*]] = fir.load %[[N_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK: %[[STEP:.*]] = arith.constant 1 : i32
  ! CHECK: omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[ARG2:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  ! CHECK: fir.store %[[ARG2]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo(%[[I_PVT_DECL]]#1, %[[A_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK: %[[NEXT_ARG2:.*]] = arith.addi %[[ARG2]], %[[STEP]] : i32
  ! CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  ! CHECK: %[[STEP_DIR:.*]] = arith.cmpi slt, %[[STEP]], %[[ZERO]] : i32
  ! CHECK: %[[LT_UB:.*]] = arith.cmpi slt, %[[NEXT_ARG2]], %[[UB]] : i32
  ! CHECK: %[[GT_UB:.*]] = arith.cmpi sgt, %[[NEXT_ARG2]], %[[UB]] : i32
  ! CHECK: %[[SEL:.*]] = arith.select %[[STEP_DIR]], %[[LT_UB]], %[[GT_UB]] : i1
  ! CHECK: fir.if %[[SEL]] {
  ! CHECK:   fir.store %[[NEXT_ARG2]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:   %[[A_PVT_LOAD:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:   hlfir.assign %[[A_PVT_LOAD]] to %[[ARG0_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:   %[[N_PVT_LOAD:.*]] = fir.load %[[N_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:   hlfir.assign %[[N_PVT_LOAD]] to %[[ARG1_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK: }

  ! CHECK: omp.yield
  ! CHECK: omp.terminator
    do i= a, n
      call foo(i, a)
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG1_DECL]]#1) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(n)
end subroutine omp_do_lastprivate2

! CHECK: func @_QPomp_do_lastprivate_collapse2(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"})
subroutine omp_do_lastprivate_collapse2(a)
  ! CHECK: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_do_lastprivate_collapse2Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer::a
  !$omp parallel do lastprivate(a) collapse(2)
  ! CHECK:  omp.parallel {

  ! CHECK: %[[A_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_do_lastprivate_collapse2Ea"}
  ! CHECK: %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse2Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[I_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
  ! CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !
  ! CHECK: %[[J_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "j", pinned, {{.*}}}
  ! CHECK: %[[J_PVT_DECL:.*]]:2 = hlfir.declare %[[J_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[LB1:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB1:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[LB2:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB2:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP2:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[ARG1:.*]], %[[ARG2:.*]]) : i32 = (%[[LB1]], %[[LB2]]) to (%[[UB1]], %[[UB2]]) inclusive step (%[[STEP1]], %[[STEP2]]) {
  ! CHECK-NEXT: fir.store %[[ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[ARG2]] to %[[J_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.call @_QPfoo(%[[I_PVT_DECL]]#1, %[[A_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK:      %[[NEXT_ARG1:.*]] = arith.addi %[[ARG1]], %[[STEP1]] : i32
  ! CHECK:      %[[ZERO1:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[STEP1_END:.*]] = arith.cmpi slt, %[[STEP1]], %[[ZERO1]] : i32
  ! CHECK:      %[[LT_UB1:.*]] = arith.cmpi slt, %[[NEXT_ARG1]], %[[UB1]] : i32
  ! CHECK:      %[[GT_UB1:.*]] = arith.cmpi sgt, %[[NEXT_ARG1]], %[[UB1]] : i32
  ! CHECK:      %[[SEL1:.*]] = arith.select %[[STEP1_END]], %[[LT_UB1]], %[[GT_UB1]] : i1
  ! CHECK:      %[[NEXT_ARG2:.*]] = arith.addi %[[ARG2]], %[[STEP2]] : i32
  ! CHECK:      %[[ZERO2:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[STEP2_END:.*]] = arith.cmpi slt, %[[STEP2]], %[[ZERO2]] : i32
  ! CHECK:      %[[LT_UB2:.*]] = arith.cmpi slt, %[[NEXT_ARG2]], %[[UB2]] : i32
  ! CHECK:      %[[GT_UB2:.*]] = arith.cmpi sgt, %[[NEXT_ARG2]], %[[UB2]] : i32
  ! CHECK:      %[[SEL2:.*]] = arith.select %[[STEP2_END]], %[[LT_UB2]], %[[GT_UB2]] : i1
  ! CHECK:      %[[AND:.*]] = arith.andi %[[SEL1]], %[[SEL2]] : i1
  ! CHECK:      fir.if %[[AND]] {
  ! CHECK:        fir.store %[[NEXT_ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:        fir.store %[[NEXT_ARG2]] to %[[J_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:        %[[A_PVT_LOAD:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:        hlfir.assign %[[A_PVT_LOAD]] to %[[ARG0_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:      }

  ! CHECK-NEXT: omp.yield
  ! CHECK-NEXT: }
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: }
    do i=1, a
      do j=1, a
        call foo(i, a)
      end do
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG0_DECL]]#1) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(a)
end subroutine omp_do_lastprivate_collapse2

! CHECK: func @_QPomp_do_lastprivate_collapse3(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"})
subroutine omp_do_lastprivate_collapse3(a)
  ! CHECK: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFomp_do_lastprivate_collapse3Ea"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer::a
  !$omp parallel do lastprivate(a) collapse(3)
  ! CHECK:  omp.parallel {

  ! CHECK: %[[A_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFomp_do_lastprivate_collapse3Ea"}
  ! CHECK: %[[A_PVT_DECL:.*]]:2 = hlfir.declare %[[A_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse3Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[I_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
  ! CHECK: %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse3Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[J_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "j", pinned, {{.*}}}
  ! CHECK: %[[J_PVT_DECL:.*]]:2 = hlfir.declare %[[J_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse3Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[K_PVT_REF:.*]] = fir.alloca i32 {bindc_name = "k", pinned, {{.*}}}
  ! CHECK: %[[K_PVT_DECL:.*]]:2 = hlfir.declare %[[K_PVT_REF]] {uniq_name = "_QFomp_do_lastprivate_collapse3Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! CHECK: %[[LB1:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB1:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[LB2:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB2:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP2:.*]] = arith.constant 1 : i32
  ! CHECK: %[[LB3:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UB3:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[STEP3:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: omp.wsloop {
  ! CHECK-NEXT: omp.loop_nest (%[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]]) : i32 = (%[[LB1]], %[[LB2]], %[[LB3]]) to (%[[UB1]], %[[UB2]], %[[UB3]]) inclusive step (%[[STEP1]], %[[STEP2]], %[[STEP3]]) {
  ! CHECK-NEXT: fir.store %[[ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[ARG2]] to %[[J_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.store %[[ARG3]] to %[[K_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK-NEXT: fir.call @_QPfoo(%[[I_PVT_DECL]]#1, %[[A_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
  ! CHECK:      %[[NEXT_ARG1:.*]] = arith.addi %[[ARG1]], %[[STEP1]] : i32
  ! CHECK:      %[[ZERO1:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[STEP1_END:.*]] = arith.cmpi slt, %[[STEP1]], %[[ZERO1]] : i32
  ! CHECK:      %[[LT_UB1:.*]] = arith.cmpi slt, %[[NEXT_ARG1]], %[[UB1]] : i32
  ! CHECK:      %[[GT_UB1:.*]] = arith.cmpi sgt, %[[NEXT_ARG1]], %[[UB1]] : i32
  ! CHECK:      %[[SEL1:.*]] = arith.select %[[STEP1_END]], %[[LT_UB1]], %[[GT_UB1]] : i1
  ! CHECK:      %[[NEXT_ARG2:.*]] = arith.addi %[[ARG2]], %[[STEP2]] : i32
  ! CHECK:      %[[ZERO2:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[STEP2_END:.*]] = arith.cmpi slt, %[[STEP2]], %[[ZERO2]] : i32
  ! CHECK:      %[[LT_UB2:.*]] = arith.cmpi slt, %[[NEXT_ARG2]], %[[UB2]] : i32
  ! CHECK:      %[[GT_UB2:.*]] = arith.cmpi sgt, %[[NEXT_ARG2]], %[[UB2]] : i32
  ! CHECK:      %[[SEL2:.*]] = arith.select %[[STEP2_END]], %[[LT_UB2]], %[[GT_UB2]] : i1
  ! CHECK:      %[[AND1:.*]] = arith.andi %[[SEL1]], %[[SEL2]] : i1
  ! CHECK:      %[[NEXT_ARG3:.*]] = arith.addi %[[ARG3]], %[[STEP3]] : i32
  ! CHECK:      %[[ZERO3:.*]] = arith.constant 0 : i32
  ! CHECK:      %[[STEP3_END:.*]] = arith.cmpi slt, %[[STEP3]], %[[ZERO3]] : i32
  ! CHECK:      %[[LT_UB3:.*]] = arith.cmpi slt, %[[NEXT_ARG3]], %[[UB3]] : i32
  ! CHECK:      %[[GT_UB3:.*]] = arith.cmpi sgt, %[[NEXT_ARG3]], %[[UB3]] : i32
  ! CHECK:      %[[SEL3:.*]] = arith.select %[[STEP3_END]], %[[LT_UB3]], %[[GT_UB3]] : i1
  ! CHECK:      %[[AND2:.*]] = arith.andi %[[AND1]], %[[SEL3]] : i1
  ! CHECK:      fir.if %[[AND2]] {
  ! CHECK:        fir.store %[[NEXT_ARG1]] to %[[I_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:        fir.store %[[NEXT_ARG2]] to %[[J_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:        fir.store %[[NEXT_ARG3]] to %[[K_PVT_DECL]]#1 : !fir.ref<i32>
  ! CHECK:        %[[A_PVT_LOAD:.*]] = fir.load %[[A_PVT_DECL]]#0 : !fir.ref<i32>
  ! CHECK:        hlfir.assign %[[A_PVT_LOAD]] to %[[ARG0_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:      }

  ! CHECK-NEXT: omp.yield
  ! CHECK-NEXT: }
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: }
    do i=1, a
      do j=1, a
        do k=1, a
          call foo(i, a)
        end do
      end do
    end do
  !$omp end parallel do
  !CHECK: fir.call @_QPbar(%[[ARG0_DECL]]#1) {{.*}}: (!fir.ref<i32>) -> ()
  call bar(a)
end subroutine omp_do_lastprivate_collapse3
