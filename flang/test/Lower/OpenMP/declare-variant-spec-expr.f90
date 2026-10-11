! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s --implicit-check-not='fir.call @_QMspec_contextPparallel'

module spec_context
contains
  ! Specification expressions are lowered before any current evaluation.
  ! The caller comes first to avoid reusing a previous procedure's evaluation.
  ! CHECK-LABEL: func.func @_QMspec_contextPfirst(
  ! CHECK-NOT: fir.call @_QMspec_contextPbound(
  ! CHECK: fir.call @_QMspec_contextPreplacement(
  ! CHECK-NOT: fir.call @_QMspec_contextPreplacement(
  ! CHECK: fir.call @_QMspec_contextPunmatched(
  ! CHECK-NOT: fir.call @_QMspec_contextPbound(
  ! CHECK: return
  subroutine first(n)
    integer, intent(in) :: n
    integer :: a(bound(n)), b(unmatched(n))
    print *, size(a), size(b)
  end subroutine

  pure integer function bound(n)
    integer, intent(in) :: n
    !$omp declare variant(replacement) match(implementation={vendor(llvm)})
    bound = n
  end function

  pure integer function unmatched(n)
    integer, intent(in) :: n
    !$omp declare variant(parallel_replacement) match(construct={parallel})
    unmatched = n
  end function

  pure integer function replacement(n)
    integer, intent(in) :: n
    replacement = n
  end function

  pure integer function parallel_replacement(n)
    integer, intent(in) :: n
    parallel_replacement = n
  end function

  ! A later procedure's prologue also starts with an empty construct context.
  ! CHECK-LABEL: func.func @_QMspec_contextPlast(
  ! CHECK-NOT: fir.call @_QMspec_contextPbound(
  ! CHECK: fir.call @_QMspec_contextPreplacement(
  ! CHECK-NOT: fir.call @_QMspec_contextPreplacement(
  ! CHECK: fir.call @_QMspec_contextPunmatched(
  ! CHECK-NOT: fir.call @_QMspec_contextPbound(
  ! CHECK: return
  subroutine last(n)
    integer, intent(in) :: n
    integer :: a(bound(n)), b(unmatched(n))
    print *, size(a), size(b)
  end subroutine
end module
