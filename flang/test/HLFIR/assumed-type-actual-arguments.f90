! Test lowering of call statements to HLFIR with assumed types
! arguments. These are a bit special because semantics do not represent
! assumed types actual arguments with an evaluate::Expr like for usual
! arguments.
! RUN: bbc -emit-hlfir --polymorphic-type -o - %s | FileCheck %s

subroutine test1(x)
  type(*) :: x
  interface
    subroutine fun1(x)
      type(*) :: x
    end subroutine
  end interface
  call fun1(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<none> {fir.bindc_name = "x"}) {
! CHECK:   %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFtest1Ex"} : (!fir.ref<none>) -> (!fir.ref<none>, !fir.ref<none>)
! CHECK:   fir.call @_QPfun1(%[[VAL_0]]#1) fastmath<contract> : (!fir.ref<none>) -> ()
! CHECK:   return
! CHECK: }

subroutine test2(x)
  type(*) :: x
  interface
    subroutine fun2(x)
      type(*) :: x(:)
    end subroutine
  end interface
  call fun2(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<none> {fir.bindc_name = "x"}) {
! CHECK:   %[[VAL_0:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFtest2Ex"} : (!fir.ref<none>) -> (!fir.ref<none>, !fir.ref<none>)
! CHECK:   %[[VAL_1:.*]] = fir.embox %[[VAL_0]]#0 : (!fir.ref<none>) -> !fir.box<none>
! CHECK:   %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.box<none>) -> !fir.box<!fir.array<?xnone>>
! CHECK:   fir.call @_QPfun2(%[[VAL_2]]) fastmath<contract> : (!fir.box<!fir.array<?xnone>>) -> ()
! CHECK:   return
! CHECK: }