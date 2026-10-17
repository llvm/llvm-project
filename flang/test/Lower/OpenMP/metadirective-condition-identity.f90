! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -fopenmp \
! RUN:   -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s

! All scores are 5/5 unless explicitly noted. Source condition identity
! determines whether a subset tie-break can override lexical order.

! CHECK-LABEL: func.func @_QPdistinct_dynamic(
! CHECK-SAME: %[[ARGA:[^:]+]]:
! CHECK: %[[A:[^:]+]]:2 = hlfir.declare %[[ARGA]]
! CHECK: %[[B:[^:]+]]:2 = hlfir.declare
! CHECK: %[[VAL:[^ ]+]] = fir.load %[[A]]#0
! CHECK: %[[COND:[^ ]+]] = fir.convert %[[VAL]]
! CHECK: fir.if %[[COND]] {
! CHECK-NEXT: omp.taskyield
! CHECK-NEXT: } else {
! CHECK: fir.load %[[B]]#0
! CHECK: fir.if
! CHECK-NEXT: omp.taskwait
! CHECK: return
subroutine distinct_dynamic(a, b)
  logical, intent(in) :: a, b
  !$omp metadirective &
  !$omp& when(user={condition(a)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(b)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPidentical_dynamic(
! CHECK-SAME: %[[ARGA:[^:]+]]:
! CHECK: %[[A:[^:]+]]:2 = hlfir.declare %[[ARGA]]
! CHECK: %[[VAL:[^ ]+]] = fir.load %[[A]]#0
! CHECK: %[[COND:[^ ]+]] = fir.convert %[[VAL]]
! CHECK: fir.if %[[COND]] {
! CHECK-NEXT: omp.taskwait
! CHECK-NEXT: } else {
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine identical_dynamic(a, b)
  logical, intent(in) :: a, b
  !$omp metadirective &
  !$omp& when(user={condition(a)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(a)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPhigher_score(
! CHECK-SAME: %[[ARGA:[^:]+]]:
! CHECK: %[[A:[^:]+]]:2 = hlfir.declare %[[ARGA]]
! CHECK: %[[B:[^:]+]]:2 = hlfir.declare
! CHECK: %[[VAL:[^ ]+]] = fir.load %[[A]]#0
! CHECK: %[[COND:[^ ]+]] = fir.convert %[[VAL]]
! CHECK: fir.if %[[COND]] {
! CHECK-NEXT: omp.taskyield
! CHECK-NEXT: } else {
! CHECK: fir.load %[[B]]#0
! CHECK: fir.if
! CHECK-NEXT: omp.taskwait
! CHECK: return
subroutine higher_score(a, b)
  logical, intent(in) :: a, b
  !$omp metadirective &
  !$omp& when(user={condition(a)}, &
  !$omp& implementation={vendor(score(5): llvm)}: taskyield) &
  !$omp& when(user={condition(b)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPdistinct_folded(
! CHECK-NOT: omp.taskwait
! CHECK: omp.taskyield
! CHECK-NOT: omp.taskwait
! CHECK: return
subroutine distinct_folded()
  !$omp metadirective &
  !$omp& when(user={condition(1 == 1)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(2 == 2)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPequivalent_spelling(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine equivalent_spelling()
  !$omp metadirective &
  !$omp& when(user={condition(1 == 1)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(1 .eq. 1)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPdistinct_sections(
! CHECK-NOT: omp.taskwait
! CHECK: omp.taskyield
! CHECK-NOT: omp.taskwait
! CHECK: return
subroutine distinct_sections()
  integer :: a(3)
  !$omp metadirective &
  !$omp& when(user={condition(size(a(1:)) > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(size(a(:1)) > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPidentical_sections(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine identical_sections()
  integer :: a(3)
  !$omp metadirective &
  !$omp& when(user={condition(size(a(1:)) > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(size(a(1:)) > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPdistinct_inquiries(
! CHECK-NOT: omp.taskwait
! CHECK: omp.taskyield
! CHECK-NOT: omp.taskwait
! CHECK: return
subroutine distinct_inquiries()
  character(3), parameter :: text(1) = ["abc"]
  !$omp metadirective &
  !$omp& when(user={condition(text(1)(:)%len > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(text(1)(:)%kind > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPidentical_inquiries(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine identical_inquiries()
  character(3), parameter :: text(1) = ["abc"]
  !$omp metadirective &
  !$omp& when(user={condition(text(1)(:)%kind > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(text(1)(:)%kind > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPplaceholder_identity(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine placeholder_identity()
  integer :: x
  !$omp metadirective &
  !$omp& when(user={condition(x%kind == 4)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(x%kind == 4)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPsame_implied_do(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine same_implied_do()
  integer(8) :: i
  !$omp metadirective &
  !$omp& when(user={condition(kind([(i, i=1,1)]) > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(kind([(i, i=1,1)]) > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPdistinct_match_any(
! CHECK-NOT: omp.taskwait
! CHECK: omp.taskyield
! CHECK-NOT: omp.taskwait
! CHECK: return
subroutine distinct_match_any(a, b)
  logical :: a, b
  !$omp metadirective &
  !$omp& when(user={condition(score(0): a)}, &
  !$omp& implementation={vendor(score(4): llvm), &
  !$omp& extension(match_any)}: taskyield) &
  !$omp& when(user={condition(score(0): b)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm), &
  !$omp& extension(match_any)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPsame_match_any(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine same_match_any(a, b)
  logical :: a, b
  !$omp metadirective &
  !$omp& when(user={condition(score(0): a)}, &
  !$omp& implementation={vendor(score(4): llvm), &
  !$omp& extension(match_any)}: taskyield) &
  !$omp& when(user={condition(score(0): a)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm), &
  !$omp& extension(match_any)}: taskwait)
end subroutine


! CHECK-LABEL: func.func @_QPrenamed_implied_do(
! CHECK-NOT: omp.taskyield
! CHECK: omp.taskwait
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine renamed_implied_do()
  integer(8) :: i, j
  !$omp metadirective &
  !$omp& when(user={condition(kind([(i, i=1,1)]) > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(kind([(j, j=1,1)]) > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine

! CHECK-LABEL: func.func @_QPdifferent_implied_do_kind(
! CHECK-NOT: omp.taskwait
! CHECK: omp.taskyield
! CHECK-NOT: omp.taskwait
! CHECK: return
subroutine different_implied_do_kind()
  integer(4) :: i
  integer(8) :: j
  !$omp metadirective &
  !$omp& when(user={condition(kind([(i, i=1,1)]) > 0)}, &
  !$omp& implementation={vendor(score(4): llvm)}: taskyield) &
  !$omp& when(user={condition(kind([(j, j=1,1)]) > 0)}, device={kind(cpu)}, &
  !$omp& implementation={vendor(score(3): llvm)}: taskwait)
end subroutine
