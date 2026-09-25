! RUN: split-file %s %t
! RUN: cd %t && %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only settings.f90
! RUN: cd %t && %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir variants.f90 -o - | FileCheck %s --check-prefix=LOCAL
! RUN: cd %t && %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir use.f90 -o - | FileCheck %s --check-prefix=IMPORTED
! RUN: cd %t && %flang_fc1 -fopenmp -fopenmp-version=52 -emit-fir use.f90 -o - | FileCheck %s --check-prefix=IMPORTED

! Identical conditions make the CPU selector a strict superset, zeroing the
! high-scoring variant. Distinct expressions and declarations must retain the
! high score even when they fold to the same value.
! LOCAL-LABEL: func.func @_QMconditionsPlocal_calls(
! LOCAL: fir.call @_QMconditionsPlow()
! LOCAL: fir.call @_QMconditionsPlow()
! LOCAL: fir.call @_QMconditionsPlow()
! LOCAL: fir.call @_QMconditionsPhigh()
! LOCAL: fir.call @_QMconditionsPhigh()
! LOCAL: fir.call @_QMconditionsPhigh()
! LOCAL: fir.call @_QMconditionsPhigh()
! LOCAL: return

! Conditions referring to USE-associated names are rebuilt in the importing
! compilation, using the loaded symbols rather than serialized pointers.
! IMPORTED-LABEL: func.func @_QPimported_conditions(
! IMPORTED: omp.taskwait
! IMPORTED: omp.taskwait
! IMPORTED: omp.taskyield
! IMPORTED: return

! Same runtime condition: the low-scoring strict superset wins when true.
! IMPORTED-LABEL: func.func @_QPruntime_same(
! IMPORTED: fir.if
! IMPORTED: omp.taskwait
! IMPORTED: } else {
! IMPORTED: return

! Different variables must not be merged even when their values might agree.
! IMPORTED-LABEL: func.func @_QPruntime_distinct(
! IMPORTED: fir.if
! IMPORTED: omp.taskyield
! IMPORTED: } else {
! IMPORTED: fir.if
! IMPORTED: omp.taskwait
! IMPORTED: return

! False conditions retain the same identity under match_none.
! IMPORTED-LABEL: func.func @_QPfalse_conditions(
! IMPORTED-NOT: omp.taskyield
! IMPORTED: omp.taskwait
! IMPORTED-NOT: omp.taskyield
! IMPORTED: return

! Both runtime outcomes under match_any retain condition identity, even when
! the implementation selector makes each candidate applicable independently.
! IMPORTED-LABEL: func.func @_QPruntime_any(
! IMPORTED: fir.if
! IMPORTED: omp.taskwait
! IMPORTED: } else {
! IMPORTED: omp.taskwait
! IMPORTED: return

!--- settings.f90
module settings
  logical, parameter :: on = .true., other = .true.
  logical :: flag, other_flag
end module

!--- variants.f90
module conditions
  use settings, only: on, on2 => on, other
contains
  integer function high()
    high = 100
  end function
  integer function low()
    low = 1
  end function

  integer function parentheses()
    !$omp declare variant(high) match(user={condition(.true.)}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition(((.true.)))}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    parentheses = 0
  end function

  integer function aliases()
    !$omp declare variant(high) match(user={condition(on)}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition(on2)}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    aliases = 0
  end function

  integer function compound_aliases()
    !$omp declare variant(high) match(user={condition(on .and. .true.)}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition((on2.and..true.))}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    compound_aliases = 0
  end function

  integer function distinct()
    !$omp declare variant(high) match(user={condition(on)}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition(other)}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    distinct = 0
  end function

  integer function literal_and_name()
    !$omp declare variant(high) match(user={condition(.true.)}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition(on)}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    literal_and_name = 0
  end function

  integer function ordered_expression()
    !$omp declare variant(high) match(user={condition(on .and. other)}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition(other .and. on)}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    ordered_expression = 0
  end function

  integer function character_literals()
    !$omp declare variant(high) match(user={condition('a b' == 'a b')}, &
    !$omp& implementation={vendor(score(100): llvm)})
    !$omp declare variant(low) match(user={condition('ab' == 'ab')}, &
    !$omp& implementation={vendor(score(1): llvm)}, device={kind(cpu)})
    character_literals = 0
  end function

  subroutine local_calls(a)
    integer :: a(7)
    a(1) = parentheses()
    a(2) = aliases()
    a(3) = compound_aliases()
    a(4) = distinct()
    a(5) = literal_and_name()
    a(6) = ordered_expression()
    a(7) = character_literals()
  end subroutine
end module

!--- use.f90
subroutine imported_conditions()
  use settings, only: on, alias => on, other
  !$omp metadirective &
  !$omp& when(user={condition(on)}, &
  !$omp& implementation={vendor(score(100): llvm)}: taskyield) &
  !$omp& when(user={condition((alias))}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(1): llvm)}: taskwait)
  !$omp metadirective &
  !$omp& when(user={condition(.true.)}, &
  !$omp& implementation={vendor(score(100): llvm)}: taskyield) &
  !$omp& when(user={condition((.true.))}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(1): llvm)}: taskwait)
  !$omp metadirective &
  !$omp& when(user={condition(on)}, &
  !$omp& implementation={vendor(score(100): llvm)}: taskyield) &
  !$omp& when(user={condition(other)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(1): llvm)}: taskwait)
end subroutine

subroutine runtime_same()
  use settings, only: flag, alias => flag
  !$omp metadirective &
  !$omp& when(user={condition(flag)}, &
  !$omp& implementation={vendor(score(100): llvm)}: taskyield) &
  !$omp& when(user={condition((alias))}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(1): llvm)}: taskwait)
end subroutine

subroutine runtime_distinct()
  use settings, only: flag, other_flag
  !$omp metadirective &
  !$omp& when(user={condition(flag)}, &
  !$omp& implementation={vendor(score(100): llvm)}: taskyield) &
  !$omp& when(user={condition(other_flag)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(1): llvm)}: taskwait)
end subroutine

subroutine false_conditions()
  !$omp metadirective &
  !$omp& when(user={condition(.false.)}, &
  !$omp& implementation={extension(match_none), vendor(score(100): gnu)}: taskyield) &
  !$omp& when(user={condition((.false.))}, device={kind(gpu)}, &
  !$omp& implementation={extension(match_none), vendor(score(1): gnu)}: taskwait)
end subroutine

subroutine runtime_any()
  use settings, only: flag, alias => flag
  !$omp metadirective &
  !$omp& when(user={condition(flag)}, &
  !$omp& implementation={extension(match_any), vendor(score(100): llvm)}: taskyield) &
  !$omp& when(user={condition((alias))}, device={kind(cpu)}, &
  !$omp& implementation={extension(match_any), vendor(score(1): llvm)}: taskwait)
end subroutine
