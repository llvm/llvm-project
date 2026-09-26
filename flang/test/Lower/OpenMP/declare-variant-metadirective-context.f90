! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-fir %s -o - | \
! RUN:   FileCheck %s

module selected_context
contains
  pure integer function in_parallel(n)
    integer, intent(in) :: n
    in_parallel = n + 1
  end function

  pure integer function in_do(n)
    integer, intent(in) :: n
    in_do = n + 2
  end function

  pure integer function value(n)
    integer, intent(in) :: n
    !$omp declare variant(in_parallel) match(construct={parallel})
    !$omp declare variant(in_do) match(construct={do})
    value = n
  end function

  ! NUM_THREADS is evaluated outside PARALLEL for either spelling. The body
  ! sees PARALLEL, and leaving the region restores the enclosing context.
  ! CHECK-LABEL: func.func @_QMselected_contextPdirect_parallel(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMselected_contextPin_parallel(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine direct_parallel(n, a)
    integer :: n, a(n)
    !$omp parallel num_threads(value(n))
      a(1) = value(n)
    !$omp end parallel
    a(n) = value(n)
  end subroutine

  ! CHECK-LABEL: func.func @_QMselected_contextPstatic_parallel(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMselected_contextPin_parallel(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine static_parallel(n, a)
    integer :: n, a(n)
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: parallel num_threads(value(n))) &
    !$omp& otherwise(nothing)
      a(1) = value(n)
    !$omp end metadirective
    a(n) = value(n)
  end subroutine

  ! Each runtime arm uses its own context. NOTHING adds no construct.
  ! CHECK-LABEL: func.func @_QMselected_contextPruntime_parallel(
  ! CHECK: fir.if
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMselected_contextPin_parallel(
  ! CHECK: } else {
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine runtime_parallel(flag, n, a)
    logical :: flag
    integer :: n, a(n)
    !$omp begin metadirective &
    !$omp& when(user={condition(flag)}: parallel num_threads(value(n))) &
    !$omp& otherwise(nothing)
      a(1) = value(n)
    !$omp end metadirective
    a(n) = value(n)
  end subroutine

  ! A loop's bounds exclude its own DO construct, while its body includes it.
  ! CHECK-LABEL: func.func @_QMselected_contextPdirect_do(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMselected_contextPin_do(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine direct_do(n, a)
    integer :: n, a(n), i
    !$omp do
    do i = 1, value(n)
      a(i) = value(n)
    end do
    a(n) = value(n)
  end subroutine

  ! The following DO remains a sibling of the standalone metadirective in
  ! the source tree, but its body must still see the selected construct.
  ! CHECK-LABEL: func.func @_QMselected_contextPstatic_do(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMselected_contextPin_do(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine static_do(n, a)
    integer :: n, a(n), i
    !$omp metadirective when(implementation={vendor(llvm)}: do) &
    !$omp& otherwise(nothing)
    do i = 1, value(n)
      a(i) = value(n)
    end do
    a(n) = value(n)
  end subroutine

  ! CHECK-LABEL: func.func @_QMselected_contextPruntime_do(
  ! CHECK: fir.if
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMselected_contextPin_do(
  ! CHECK: } else {
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine runtime_do(flag, n, a)
    logical :: flag
    integer :: n, a(n), i
    !$omp metadirective when(user={condition(flag)}: do) otherwise(nothing)
    do i = 1, value(n)
      a(i) = value(n)
    end do
    a(n) = value(n)
  end subroutine

  ! The delimited form owns the loop in the source tree.
  ! CHECK-LABEL: func.func @_QMselected_contextPdelimited_do(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMselected_contextPin_do(
  ! CHECK: fir.call @_QMselected_contextPvalue(
  ! CHECK: return
  subroutine delimited_do(n, a)
    integer :: n, a(n), i
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: do) otherwise(nothing)
    do i = 1, value(n)
      a(i) = value(n)
    end do
    !$omp end metadirective
    a(n) = value(n)
  end subroutine
end module
