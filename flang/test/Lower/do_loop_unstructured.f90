! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fwrapv -o - %s | FileCheck %s --check-prefix=NO-NSW

! Tests for unstructured loops.

! NO-NSW-NOT: overflow<nsw>

! Test a simple unstructured loop. Test for the existence of,
! -> The initialization of the trip-count and loop-variable
! -> The branch to the body or the exit inside the header
! -> The increment of the trip-count and the loop-variable inside the body
subroutine simple_unstructured()
  integer :: i
  do i=1,100
    goto 404
    404 continue
  end do
end subroutine
! CHECK-LABEL: simple_unstructured
! CHECK:   %[[TRIP_VAR_REF:.*]] = fir.alloca i32
! CHECK:   %[[LOOP_VAR_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_unstructuredEi"}
! CHECK:   %[[LOOP_VAR_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_REF]]
! CHECK:   %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:   %[[HUNDRED:.*]] = arith.constant 100 : i32
! CHECK:   %[[STEP_ONE:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[HUNDRED]], %[[ONE]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[STEP_ONE]] : i32
! CHECK:   %[[TRIP_COUNT:.*]] = arith.divsi %[[TMP2]], %[[STEP_ONE]] : i32
! CHECK:   fir.store %[[TRIP_COUNT]] to %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[ONE]] to %[[LOOP_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER:.*]]
! CHECK: ^[[HEADER]]:
! CHECK:   %[[TRIP_VAR:.*]] = fir.load %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND:.*]] = arith.cmpi sgt, %[[TRIP_VAR]], %[[ZERO]] : i32
! CHECK:   cf.cond_br %[[COND]], ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK: ^[[BODY]]:
! CHECK:   %[[TRIP_VAR:.*]] = fir.load %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[ONE_1:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_NEXT:.*]] = arith.subi %[[TRIP_VAR]], %[[ONE_1]] : i32
! CHECK:   fir.store %[[TRIP_VAR_NEXT]] to %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR:.*]] = fir.load %[[LOOP_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[STEP_ONE_2:.*]] = arith.constant 1 : i32
! CHECK:   %[[LOOP_VAR_NEXT:.*]] = arith.addi %[[LOOP_VAR]], %[[STEP_ONE_2]] overflow<nsw> : i32
! CHECK:   fir.store %[[LOOP_VAR_NEXT]] to %[[LOOP_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER]]
! CHECK: ^[[EXIT]]:
! CHECK:   return

! Test an unstructured loop with a step. Mostly similar to the previous one.
! Only difference is a non-unit step.
subroutine simple_unstructured_with_step()
  integer :: i
  do i=1,100,2
    goto 404
    404 continue
  end do
end subroutine
! CHECK-LABEL: simple_unstructured_with_step
! CHECK:   %[[TRIP_VAR_REF:.*]] = fir.alloca i32
! CHECK:   %[[LOOP_VAR_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_unstructured_with_stepEi"}
! CHECK:   %[[LOOP_VAR_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_REF]]
! CHECK:   %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:   %[[HUNDRED:.*]] = arith.constant 100 : i32
! CHECK:   %[[STEP:.*]] = arith.constant 2 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[HUNDRED]], %[[ONE]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[STEP]] : i32
! CHECK:   %[[TRIP_COUNT:.*]] = arith.divsi %[[TMP2]], %[[STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT]] to %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[ONE]] to %[[LOOP_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER:.*]]
! CHECK: ^[[HEADER]]:
! CHECK:   %[[TRIP_VAR:.*]] = fir.load %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND:.*]] = arith.cmpi sgt, %[[TRIP_VAR]], %[[ZERO]] : i32
! CHECK:   cf.cond_br %[[COND]], ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK: ^[[BODY]]:
! CHECK:   %[[TRIP_VAR:.*]] = fir.load %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[ONE_1:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_NEXT:.*]] = arith.subi %[[TRIP_VAR]], %[[ONE_1]] : i32
! CHECK:   fir.store %[[TRIP_VAR_NEXT]] to %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR:.*]] = fir.load %[[LOOP_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[STEP_2:.*]] = arith.constant 2 : i32
! CHECK:   %[[LOOP_VAR_NEXT:.*]] = arith.addi %[[LOOP_VAR]], %[[STEP_2]] overflow<nsw> : i32
! CHECK:   fir.store %[[LOOP_VAR_NEXT]] to %[[LOOP_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER]]
! CHECK: ^[[EXIT]]:
! CHECK:   return

! Test a three nested unstructured loop. Three nesting is the basic case where
! we have loops that are neither innermost or outermost.
subroutine nested_unstructured()
  integer :: i, j, k
  do i=1,100
    do j=1,200
      do k=1,300
        goto 404
        404 continue
      end do
    end do
  end do
end subroutine
! With the wrap-unstructured-constructs-in-execute-region pass, the innermost
! k-loop is the only one classified unstructured (the `goto 404`/`404 continue`
! pattern). It gets wrapped in scf.execute_region, and the outer i and j
! loops fold back to fir.do_loop.
! CHECK-LABEL: nested_unstructured
! CHECK:   %[[TRIP_VAR_K_REF:.*]] = fir.alloca i32
! CHECK:   %[[LOOP_VAR_I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFnested_unstructuredEi"}
! CHECK:   %[[LOOP_VAR_I_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_I_REF]]
! CHECK:   %[[LOOP_VAR_J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_unstructuredEj"}
! CHECK:   %[[LOOP_VAR_J_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_J_REF]]
! CHECK:   %[[LOOP_VAR_K_REF:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFnested_unstructuredEk"}
! CHECK:   %[[LOOP_VAR_K_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_K_REF]]
! CHECK:   fir.do_loop %{{[^ ]+}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
! CHECK:     fir.do_loop %{{[^ ]+}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
! CHECK:       scf.execute_region no_inline {
! CHECK:         cf.br ^[[HEADER_K:.*]]
! CHECK:       ^[[HEADER_K]]:
! CHECK:         %[[TRIP_COUNT_K:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
! CHECK:         fir.store %[[TRIP_COUNT_K]] to %[[TRIP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:         fir.store %{{.*}} to %[[LOOP_VAR_K_DECL]]#0 : !fir.ref<i32>
! CHECK:         cf.br ^[[HEADER_K_BODY:.*]]
! CHECK:       ^[[HEADER_K_BODY]]:
! CHECK:         %[[TRIP_VAR_K:.*]] = fir.load %[[TRIP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:         %[[COND_K:.*]] = arith.cmpi sgt, %[[TRIP_VAR_K]], %{{.*}} : i32
! CHECK:         cf.cond_br %[[COND_K]], ^[[BODY_K:.*]], ^[[EXIT_K:.*]]
! CHECK:       ^[[BODY_K]]:
! CHECK:         cf.br ^{{.*}}
! CHECK:         cf.br ^[[HEADER_K_BODY]]
! CHECK:       ^[[EXIT_K]]:
! CHECK:         scf.yield
! CHECK:       }
! CHECK:     }
! CHECK:   }
! CHECK:   return

! Test the existence of a structured loop inside an unstructured loop.
! Only minimal checks are inserted for the structured loop.
subroutine nested_structured_in_unstructured()
  integer :: i, j
  do i=1,100
    do j=1,100
    end do
    goto 404
    404 continue
  end do
end subroutine
! CHECK-LABEL: nested_structured_in_unstructured
! CHECK:   %[[TRIP_VAR_I_REF:.*]] = fir.alloca i32
! CHECK:   %[[LOOP_VAR_I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFnested_structured_in_unstructuredEi"}
! CHECK:   %[[LOOP_VAR_I_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_I_REF]]
! CHECK:   %[[LOOP_VAR_J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_structured_in_unstructuredEj"}
! CHECK:   %[[LOOP_VAR_J_DECL:.*]]:2 = hlfir.declare %[[LOOP_VAR_J_REF]]
! CHECK:   %[[I_START:.*]] = arith.constant 1 : i32
! CHECK:   %[[I_END:.*]] = arith.constant 100 : i32
! CHECK:   %[[I_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[I_END]], %[[I_START]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[I_STEP]] : i32
! CHECK:   %[[TRIP_COUNT:.*]] = arith.divsi %[[TMP2]], %[[I_STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT]] to %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[I_START]] to %[[LOOP_VAR_I_DECL]]#0 : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER:.*]]
! CHECK: ^[[HEADER]]:
! CHECK:   %[[TRIP_VAR:.*]] = fir.load %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND:.*]] = arith.cmpi sgt, %[[TRIP_VAR]], %[[ZERO]] : i32
! CHECK:   cf.cond_br %[[COND]], ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK: ^[[BODY]]:
! CHECK:   fir.do_loop %[[J_IV:[^ ]*]] =
! CHECK-SAME: %[[J_LB:[^ ]*]] to %[[J_UB:[^ ]*]] step %[[J_ST:[^ ]*]] : i32 {
! CHECK:     fir.store %[[J_IV]] to %[[LOOP_VAR_J_DECL]]#0 : !fir.ref<i32>
! CHECK:   }
! CHECK:   %[[J_LBIDX:.*]] = fir.convert %[[J_LB]] : (i32) -> index
! CHECK:   %[[J_UBIDX:.*]] = fir.convert %[[J_UB]] : (i32) -> index
! CHECK:   %[[J_STIDX:.*]] = fir.convert %[[J_ST]] : (i32) -> index
! CHECK:   %[[J_C0:.*]] = arith.constant 0 : index
! CHECK:   %[[J_DIFF:.*]] = arith.subi %[[J_UBIDX]], %[[J_LBIDX]] : index
! CHECK:   %[[J_ADD:.*]] = arith.addi %[[J_DIFF]], %[[J_STIDX]] : index
! CHECK:   %[[J_TRIP:.*]] = arith.divsi %[[J_ADD]], %[[J_STIDX]] : index
! CHECK:   %[[J_CMP:.*]] = arith.cmpi slt, %[[J_TRIP]], %[[J_C0]] : index
! CHECK:   %[[J_SEL:.*]] = arith.select %[[J_CMP]], %[[J_C0]], %[[J_TRIP]] : index
! CHECK:   %[[J_MUL:.*]] = arith.muli %[[J_SEL]], %[[J_STIDX]] : index
! CHECK:   %[[J_LASTIDX:.*]] = arith.addi %[[J_LBIDX]], %[[J_MUL]] : index
! CHECK:   %[[J_LAST:.*]] = fir.convert %[[J_LASTIDX]] : (index) -> i32
! CHECK:   fir.store %[[J_LAST]] to %[[LOOP_VAR_J_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[TRIP_VAR_I:.*]] = fir.load %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[C1_3:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_I_NEXT:.*]] = arith.subi %[[TRIP_VAR_I]], %[[C1_3]] : i32
! CHECK:   fir.store %[[TRIP_VAR_I_NEXT]] to %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_I:.*]] = fir.load %[[LOOP_VAR_I_DECL]]#0 : !fir.ref<i32>
! CHECK:   %[[I_STEP_2:.*]] = arith.constant 1 : i32
! CHECK:   %[[LOOP_VAR_I_NEXT:.*]] = arith.addi %[[LOOP_VAR_I]], %[[I_STEP_2]] overflow<nsw> : i32
! CHECK:   fir.store %[[LOOP_VAR_I_NEXT]] to %[[LOOP_VAR_I_DECL]]#0 : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER]]
! CHECK: ^[[EXIT]]:
! CHECK:   return

subroutine unstructured_do_concurrent
  logical :: success
  do concurrent (i=1:10) local(success)
    success = .false.
    error stop "fail"
  enddo
end
! CHECK-LABEL: func.func @_QPunstructured_do_concurrent
! CHECK:         %[[ITER_VAR:.*]] = fir.alloca i32
! CHECK:         scf.execute_region no_inline {
! CHECK:           cf.br ^[[HEADER:.*]]
! CHECK:         ^[[HEADER]]:
! CHECK:           %{{.*}} = fir.load %[[ITER_VAR]] : !fir.ref<i32>
! CHECK:           cf.cond_br %{{.*}}, ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK:         ^[[BODY]]:
! CHECK-NEXT:      %{{.*}} = fir.alloca !fir.logical<4> {bindc_name = "success", {{.*}}}
! CHECK:         ^[[EXIT]]:
! CHECK-NEXT:      scf.yield
! CHECK:         }
! CHECK:         return
