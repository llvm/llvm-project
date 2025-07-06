! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine TestOfCharacter(a0, a1, l)
  character(len=*), intent(in) :: a0
  character(len=*), intent(inout):: a1
  integer, intent(in) :: l

  !$omp target map(to:a0) map(from: a1)
  a1 = a0
  !$omp end target
end subroutine TestOfCharacter


!CHECK:  func.func @_QPtestofcharacter(%[[ARG0:.*]]: !fir.boxchar<1> {{.*}}, %[[ARG1:.*]]: !fir.boxchar<1> {{.*}}
!CHECK:  %[[A0_BOXCHAR_ALLOCA:.*]] = fir.alloca !fir.boxchar<1>
!CHECK:  %[[A1_BOXCHAR_ALLOCA:.*]] = fir.alloca !fir.boxchar<1>
!CHECK:  %[[UNBOXED_ARG0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:  %[[A0_DECL:.*]]:2 = hlfir.declare %[[UNBOXED_ARG0]]#0 typeparams %[[UNBOXED_ARG0]]#1 dummy_scope {{.*}} -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
!CHECK:  %[[UNBOXED_ARG1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:  %[[A1_DECL:.*]]:2 = hlfir.declare %[[UNBOXED_ARG1]]#0 typeparams %[[UNBOXED_ARG1]]#1 dummy_scope {{.*}} -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
!CHECK:  %[[A0_LB:.*]] = arith.constant 0 : index
!CHECK:  %[[A0_STRIDE:.*]] = arith.constant 1 : index
!CHECK:  %[[UNBOXED_A0_DECL:.*]]:2 = fir.unboxchar %[[A0_DECL]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:  %[[A0_UB:.*]] = arith.subi %[[UNBOXED_A0_DECL]]#1, %[[A0_STRIDE]] : index
!CHECK:  %[[A0_BOUNDS:.*]] = omp.map.bounds lower_bound(%[[A0_LB]] : index) upper_bound(%[[A0_UB]] : index) extent(%[[UNBOXED_A0_DECL]]#1 : index)
!CHECK-SAME:  stride(%[[A0_STRIDE]] : index) start_idx(%[[A0_LB]] : index) {stride_in_bytes = true}
!CHECK:  %[[A0_MAP:.*]] = omp.map.info var_ptr(%[[A0_DECL]]#1 : !fir.ref<!fir.char<1,?>>, !fir.char<1,?>) map_clauses(to) capture(ByRef) bounds(%[[A0_BOUNDS]]) -> !fir.ref<!fir.char<1,?>> {name = "a0"}
!CHECK:  %[[A1_LB:.*]] = arith.constant 0 : index
!CHECK:  %[[A1_STRIDE:.*]] = arith.constant 1 : index
!CHECK:  %[[UNBOXED_A1_DECL:.*]]:2 = fir.unboxchar %[[A1_DECL]]#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:  %[[A1_UB:.*]] = arith.subi %[[UNBOXED_A1_DECL]]#1, %[[A1_STRIDE]] : index
!CHECK:  %[[A1_BOUNDS:.*]] = omp.map.bounds lower_bound(%[[A1_LB]] : index) upper_bound(%[[A1_UB]] : index) extent(%[[UNBOXED_A1_DECL]]#1 : index)
!CHECKL-SAME: stride(%[[A1_STRIDE]] : index) start_idx(%[[A1_LB]] : index) {stride_in_bytes = true}
!CHECK:  %[[A1_MAP:.*]] = omp.map.info var_ptr(%[[A1_DECL]]#1 : !fir.ref<!fir.char<1,?>>, !fir.char<1,?>) map_clauses(from) capture(ByRef) bounds(%[[A1_BOUNDS]]) -> !fir.ref<!fir.char<1,?>> {name = "a1"}
!CHECK:  fir.store %[[ARG1]] to %[[A1_BOXCHAR_ALLOCA]] : !fir.ref<!fir.boxchar<1>>
!CHECK:  %[[CONST_ZERO:.*]] = arith.constant 0 : index
!CHECK:  %[[CONST_ONE:.*]] = arith.constant 1 : index
!CHECK: %[[UNBOXED_ARG1:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:  %[[A1_UB:.*]] = arith.subi %[[UNBOXED_ARG1]]#1, %[[CONST_ONE]] : index
!CHECK:  %[[BOUNDS_A1_BOXCHAR:.*]] = omp.map.bounds lower_bound(%[[CONST_ZERO]] : index) upper_bound(%[[A1_UB]] : index) extent(%[[UNBOXED_ARG1]]#1 : index)
!CHECK-SAME: stride(%[[CONST_ONE]] : index) start_idx(%[[CONST_ZERO]] : index) {stride_in_bytes = true}
!CHECK:  %[[A1_BOXCHAR_MAP:.*]] = omp.map.info var_ptr(%[[A1_BOXCHAR_ALLOCA]] : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>) map_clauses(implicit, to)
!CHECK-SAME: capture(ByRef) bounds(%[[BOUNDS_A1_BOXCHAR]]) -> !fir.ref<!fir.boxchar<1>> {name = ""}
!CHECK:  fir.store %[[ARG0]] to %[[A0_BOXCHAR_ALLOCA]] : !fir.ref<!fir.boxchar<1>>
!CHECK:  %[[CONST_ZERO:.*]] = arith.constant 0 : index
!CHECK:  %[[CONST_ONE:.*]] = arith.constant 1 : index
!CHECK: %[[UNBOXED_ARG0:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:  %[[A0_UB:.*]] = arith.subi %[[UNBOXED_ARG0]]#1, %[[CONST_ONE]] : index
!CHECK:  %[[BOUNDS_A0_BOXCHAR:.*]] = omp.map.bounds lower_bound(%[[CONST_ZERO]] : index) upper_bound(%[[A0_UB]] : index) extent(%[[UNBOXED_ARG0]]#1 : index)
!CHECK-SAME: stride(%[[CONST_ONE]] : index) start_idx(%[[CONST_ZERO]] : index) {stride_in_bytes = true}
!CHECK:  %[[A0_BOXCHAR_MAP:.*]] = omp.map.info var_ptr(%[[A0_BOXCHAR_ALLOCA]] : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>) map_clauses(implicit, to)
!CHECK-SAME: capture(ByRef) bounds(%[[BOUNDS_A0_BOXCHAR]]) -> !fir.ref<!fir.boxchar<1>> {name = ""}
!CHECK:  omp.target map_entries(%[[A0_MAP]] -> %[[TGT_A0:.*]], %[[A1_MAP]] -> %[[TGT_A1:.*]], %[[A1_BOXCHAR_MAP]] -> %[[TGT_A1_BOXCHAR:.*]], %[[A0_BOXCHAR_MAP]] -> %[[TGT_A0_BOXCHAR:.*]] : !fir.ref<!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>, !fir.ref<!fir.boxchar<1>>, !fir.ref<!fir.boxchar<1>>) {
!CHECK:    %[[TGT_A0_BC_LD:.*]] = fir.load %[[TGT_A0_BOXCHAR]] : !fir.ref<!fir.boxchar<1>>
!CHECK:    %[[TGT_A1_BC_LD:.*]] = fir.load %[[TGT_A1_BOXCHAR]] : !fir.ref<!fir.boxchar<1>>
!CHECK:    %[[UNBOXED_TGT_A1:.*]]:2 = fir.unboxchar %[[TGT_A1_BC_LD]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:    %[[UNBOXED_TGT_A0:.*]]:2 = fir.unboxchar %[[TGT_A0_BC_LD]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
!CHECK:    %[[TGT_A0_DECL:.*]]:2 = hlfir.declare %[[TGT_A0]] typeparams %[[UNBOXED_TGT_A0]]#1 {{.*}} -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
!CHECK:    %[[TGT_A1_DECL:.*]]:2 = hlfir.declare %[[TGT_A1]] typeparams %[[UNBOXED_TGT_A1]]#1 {{.*}} -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)

