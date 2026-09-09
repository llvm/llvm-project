! RUN: %flang_fc1 -emit-hlfir -mmlir --wrap-unstructured-constructs-in-execute-region -o - %s | \
! RUN:   FileCheck %s --implicit-check-not=scf.execute_region

! An assigned GO TO may branch to any non-FORMAT label that has been ASSIGN'd to
! the variable, even one absent from the statement's explicit label list; see the
! relaxation in genFIR(AssignedGotoStmt) in flang/lib/Lower/Bridge.cpp.  Such a
! target can sit inside a DO construct, as label 20 does here, so the construct
! must not be wrapped in scf.execute_region -- the branch would then cross a
! region boundary and the fir.select would fail verification with "branching to
! block of a different region".

subroutine assigned_goto_into_loop(a, n)
  integer :: n, i, k
  real :: a(n)
  do i = 1, n
     assign 20 to k
20   a(i) = a(i) + 1.0
  end do
  go to k, (30, 40)
30 a(1) = 0.0
40 continue
end subroutine

! CHECK-LABEL: func.func @_QPassigned_goto_into_loop(
! The switch admits label 20, which is ASSIGN'd but absent from the (30, 40)
! list.  Compiling at all shows its target block is in the branch's own region.
! CHECK: fir.select %{{.*}} : i32 [20, ^bb{{[0-9]+}}, unit, ^bb{{[0-9]+}}]
