! RUN: bbc -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s

! Tests for unstructured loops.

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
! CHECK:   %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:   %[[HUNDRED:.*]] = arith.constant 100 : i32
! CHECK:   %[[STEP_ONE:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[HUNDRED]], %[[ONE]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[STEP_ONE]] : i32
! CHECK:   %[[TRIP_COUNT:.*]] = arith.divsi %[[TMP2]], %[[STEP_ONE]] : i32
! CHECK:   fir.store %[[TRIP_COUNT]] to %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[ONE]] to %[[LOOP_VAR_REF]] : !fir.ref<i32>
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
! CHECK:   %[[LOOP_VAR:.*]] = fir.load %[[LOOP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_NEXT:.*]] = arith.addi %[[LOOP_VAR]], %[[STEP_ONE]] : i32
! CHECK:   fir.store %[[LOOP_VAR_NEXT]] to %[[LOOP_VAR_REF]] : !fir.ref<i32>
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
! CHECK:   %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:   %[[HUNDRED:.*]] = arith.constant 100 : i32
! CHECK:   %[[STEP:.*]] = arith.constant 2 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[HUNDRED]], %[[ONE]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[STEP]] : i32
! CHECK:   %[[TRIP_COUNT:.*]] = arith.divsi %[[TMP2]], %[[STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT]] to %[[TRIP_VAR_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[ONE]] to %[[LOOP_VAR_REF]] : !fir.ref<i32>
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
! CHECK:   %[[LOOP_VAR:.*]] = fir.load %[[LOOP_VAR_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_NEXT:.*]] = arith.addi %[[LOOP_VAR]], %[[STEP]] : i32
! CHECK:   fir.store %[[LOOP_VAR_NEXT]] to %[[LOOP_VAR_REF]] : !fir.ref<i32>
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
! CHECK-LABEL: nested_unstructured
! CHECK:   %[[TRIP_VAR_K_REF:.*]] = fir.alloca i32
! CHECK:   %[[TRIP_VAR_J_REF:.*]] = fir.alloca i32
! CHECK:   %[[TRIP_VAR_I_REF:.*]] = fir.alloca i32
! CHECK:   %[[LOOP_VAR_I_REF:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFnested_unstructuredEi"}
! CHECK:   %[[LOOP_VAR_J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_unstructuredEj"}
! CHECK:   %[[LOOP_VAR_K_REF:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFnested_unstructuredEk"}
! CHECK:   %[[I_START:.*]] = arith.constant 1 : i32
! CHECK:   %[[I_END:.*]] = arith.constant 100 : i32
! CHECK:   %[[I_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[I_END]], %[[I_START]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[I_STEP]] : i32
! CHECK:   %[[TRIP_COUNT_I:.*]] = arith.divsi %[[TMP2]], %[[I_STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT_I]] to %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[I_START]] to %[[LOOP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER_I:.*]]
! CHECK: ^[[HEADER_I]]:
! CHECK:   %[[TRIP_VAR_I:.*]] = fir.load %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO_1:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND_I:.*]] = arith.cmpi sgt, %[[TRIP_VAR_I]], %[[ZERO_1]] : i32
! CHECK:   cf.cond_br %[[COND_I]], ^[[BODY_I:.*]], ^[[EXIT_I:.*]]
! CHECK: ^[[BODY_I]]:
! CHECK:   %[[J_START:.*]] = arith.constant 1 : i32
! CHECK:   %[[J_END:.*]] = arith.constant 200 : i32
! CHECK:   %[[J_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP3:.*]] = arith.subi %[[J_END]], %[[J_START]] : i32
! CHECK:   %[[TMP4:.*]] = arith.addi %[[TMP3]], %[[J_STEP]] : i32
! CHECK:   %[[TRIP_COUNT_J:.*]] = arith.divsi %[[TMP4]], %[[J_STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT_J]] to %[[TRIP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[J_START]] to %[[LOOP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER_J:.*]]
! CHECK: ^[[HEADER_J]]:
! CHECK:   %[[TRIP_VAR_J:.*]] = fir.load %[[TRIP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO_2:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND_J:.*]] = arith.cmpi sgt, %[[TRIP_VAR_J]], %[[ZERO_2]] : i32
! CHECK:   cf.cond_br %[[COND_J]], ^[[BODY_J:.*]], ^[[EXIT_J:.*]]
! CHECK: ^[[BODY_J]]:
! CHECK:   %[[K_START:.*]] = arith.constant 1 : i32
! CHECK:   %[[K_END:.*]] = arith.constant 300 : i32
! CHECK:   %[[K_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP3:.*]] = arith.subi %[[K_END]], %[[K_START]] : i32
! CHECK:   %[[TMP4:.*]] = arith.addi %[[TMP3]], %[[K_STEP]] : i32
! CHECK:   %[[TRIP_COUNT_K:.*]] = arith.divsi %[[TMP4]], %[[K_STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT_K]] to %[[TRIP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[K_START]] to %[[LOOP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER_K:.*]]
! CHECK: ^[[HEADER_K]]:
! CHECK:   %[[TRIP_VAR_K:.*]] = fir.load %[[TRIP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO_2:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND_K:.*]] = arith.cmpi sgt, %[[TRIP_VAR_K]], %[[ZERO_2]] : i32
! CHECK:   cf.cond_br %[[COND_K]], ^[[BODY_K:.*]], ^[[EXIT_K:.*]]
! CHECK: ^[[BODY_K]]:
! CHECK:   %[[TRIP_VAR_K:.*]] = fir.load %[[TRIP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   %[[ONE_1:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_K_NEXT:.*]] = arith.subi %[[TRIP_VAR_K]], %[[ONE_1]] : i32
! CHECK:   fir.store %[[TRIP_VAR_K_NEXT]] to %[[TRIP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_K:.*]] = fir.load %[[LOOP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_K_NEXT:.*]] = arith.addi %[[LOOP_VAR_K]], %[[K_STEP]] : i32
! CHECK:   fir.store %[[LOOP_VAR_K_NEXT]] to %[[LOOP_VAR_K_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER_K]]
! CHECK: ^[[EXIT_K]]:
! CHECK:   %[[TRIP_VAR_J:.*]] = fir.load %[[TRIP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   %[[ONE_1:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_J_NEXT:.*]] = arith.subi %[[TRIP_VAR_J]], %[[ONE_1]] : i32
! CHECK:   fir.store %[[TRIP_VAR_J_NEXT]] to %[[TRIP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_J:.*]] = fir.load %[[LOOP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_J_NEXT:.*]] = arith.addi %[[LOOP_VAR_J]], %[[J_STEP]] : i32
! CHECK:   fir.store %[[LOOP_VAR_J_NEXT]] to %[[LOOP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER_J]]
! CHECK: ^[[EXIT_J]]:
! CHECK:   %[[TRIP_VAR_I:.*]] = fir.load %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[ONE_1:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_I_NEXT:.*]] = arith.subi %[[TRIP_VAR_I]], %[[ONE_1]] : i32
! CHECK:   fir.store %[[TRIP_VAR_I_NEXT]] to %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_I:.*]] = fir.load %[[LOOP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_I_NEXT:.*]] = arith.addi %[[LOOP_VAR_I]], %[[I_STEP]] : i32
! CHECK:   fir.store %[[LOOP_VAR_I_NEXT]] to %[[LOOP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER_I]]
! CHECK: ^[[EXIT_I]]:
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
! CHECK:   %[[LOOP_VAR_J_REF:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFnested_structured_in_unstructuredEj"}
! CHECK:   %[[I_START:.*]] = arith.constant 1 : i32
! CHECK:   %[[I_END:.*]] = arith.constant 100 : i32
! CHECK:   %[[I_STEP:.*]] = arith.constant 1 : i32
! CHECK:   %[[TMP1:.*]] = arith.subi %[[I_END]], %[[I_START]] : i32
! CHECK:   %[[TMP2:.*]] = arith.addi %[[TMP1]], %[[I_STEP]] : i32
! CHECK:   %[[TRIP_COUNT:.*]] = arith.divsi %[[TMP2]], %[[I_STEP]] : i32
! CHECK:   fir.store %[[TRIP_COUNT]] to %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   fir.store %[[I_START]] to %[[LOOP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER:.*]]
! CHECK: ^[[HEADER]]:
! CHECK:   %[[TRIP_VAR:.*]] = fir.load %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK:   %[[COND:.*]] = arith.cmpi sgt, %[[TRIP_VAR]], %[[ZERO]] : i32
! CHECK:   cf.cond_br %[[COND]], ^[[BODY:.*]], ^[[EXIT:.*]]
! CHECK: ^[[BODY]]:
! CHECK:   %{{.*}} = fir.do_loop %[[J_INDEX:[^ ]*]] =
! CHECK-SAME: %{{.*}} to %{{.*}} step %{{[^ ]*}}
! CHECK-SAME: iter_args(%[[J_IV:.*]] = %{{.*}}) -> (index, i32) {
! CHECK:     fir.store %[[J_IV]] to %[[LOOP_VAR_J_REF]] : !fir.ref<i32>
! CHECK:   }
! CHECK:   %[[TRIP_VAR_I:.*]] = fir.load %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[C1_3:.*]] = arith.constant 1 : i32
! CHECK:   %[[TRIP_VAR_I_NEXT:.*]] = arith.subi %[[TRIP_VAR_I]], %[[C1_3]] : i32
! CHECK:   fir.store %[[TRIP_VAR_I_NEXT]] to %[[TRIP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_I:.*]] = fir.load %[[LOOP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   %[[LOOP_VAR_I_NEXT:.*]] = arith.addi %[[LOOP_VAR_I]], %c1_i32_0 : i32
! CHECK:   fir.store %[[LOOP_VAR_I_NEXT]] to %[[LOOP_VAR_I_REF]] : !fir.ref<i32>
! CHECK:   cf.br ^[[HEADER]]
! CHECK: ^[[EXIT]]:
! CHECK:   return
