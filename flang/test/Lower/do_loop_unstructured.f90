! RUN: %flang_fc1 -emit-hlfir -mmlir --wrap-unstructured-constructs-in-execute-region -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fwrapv -mmlir --wrap-unstructured-constructs-in-execute-region -o - %s | FileCheck %s --check-prefix=NO-NSW

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
! The GOTO targets the statement that follows it, so nothing branches out of
! the body and the loop keeps its structured form.
! CHECK-LABEL: simple_unstructured
! CHECK:   %[[LOOP_VAR_REF:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFsimple_unstructuredEi"}>
! CHECK:   fir.do_loop %[[IV:.*]] = %c1{{.*}} to %c100{{.*}} step %c1{{.*}} : i32 {
! CHECK:     fir.store %[[IV]] to
! CHECK:   }
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
! Same, with an explicit step.
! CHECK-LABEL: simple_unstructured_with_step
! CHECK:   fir.do_loop %[[IV:.*]] = %c1{{.*}} to %c100{{.*}} step %c2{{.*}} : i32 {
! CHECK:     fir.store %[[IV]] to
! CHECK:   }
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
! The innermost GOTO stays inside its own body, so every level stays
! structured and no wrap is needed at all.
! CHECK-LABEL: nested_unstructured
! CHECK:   fir.do_loop %{{.*}} = %c1{{.*}} to %c100{{.*}} step %c1{{.*}} : i32 {
! CHECK:     fir.do_loop %{{.*}} = %c1{{.*}} to %c200{{.*}} step %c1{{.*}} : i32 {
! CHECK:       fir.do_loop %{{.*}} = %c1{{.*}} to %c300{{.*}} step %c1{{.*}} : i32 {
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
! The GOTO follows an inner loop, so the outer body needs raw blocks and is
! wrapped, while the inner loop stays a plain fir.do_loop inside the wrap.
! CHECK-LABEL: nested_structured_in_unstructured
! CHECK:   fir.do_loop %{{.*}} = %c1{{.*}} to %c100{{.*}} step %c1{{.*}} : i32 {
! CHECK:     scf.execute_region no_inline {
! CHECK:       fir.do_loop %{{.*}} = %c1{{.*}} to %c100{{.*}} step %c1{{.*}} : i32 {
! CHECK:       scf.yield
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
! CHECK-NEXT:      %{{.*}} = fir.alloca !fir.logical<4> <{bindc_name = "success", {{.*}}}>
! CHECK:         ^[[EXIT]]:
! CHECK-NEXT:      scf.yield
! CHECK:         }
! CHECK:         return
