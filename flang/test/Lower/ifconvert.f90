! RUN: %flang_fc1 -fdebug-dump-pft %s 2>&1 | FileCheck %s

! Note: PFT dump output is fairly stable, including node indexes and
!       annotations, so all output is CHECKed.

  ! CHECK: 1 Program <anonymous>
  ! CHECK:   1 PrintStmt: print*
  print*

  ! CHECK:   <<DoConstruct>> -> 8
  ! CHECK:     2 NonLabelDoStmt -> 7: do i = 1, 5
  ! CHECK:     <<IfConstruct>> -> 7
  ! CHECK:       3 ^IfStmt [negate] -> 7: if(i <= 1 .or. i >= 5) cycle
  ! CHECK:       6 ^PrintStmt: print*, i
  ! CHECK:       5 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     7 EndDoStmt -> 2: end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle
     print*, i
  end do

  ! CHECK:   8 PrintStmt: print*
  print*

  ! CHECK:   <<DoConstruct>> -> 15
  ! CHECK:     9 NonLabelDoStmt -> 14: do i = 1, 5
  ! CHECK:     <<IfConstruct>> -> 14
  ! CHECK:       10 ^IfStmt [negate] -> 14: if(i <= 1 .or. i >= 5) cycle
  ! CHECK:       13 ^PrintStmt: print*, i
  ! CHECK:       12 EndIfStmt
  ! CHECK:     <<End IfConstruct>>
  ! CHECK:     14 EndDoStmt -> 9: 2 end do
  ! CHECK:   <<End DoConstruct>>
  do i = 1, 5
     if (i <= 1 .or. i >= 5) cycle
     print*, i
2 end do

  ! CHECK:   15 PrintStmt: print*
  print*

  ! CHECK:<<DoConstruct!>> -> 30
  ! CHECK:  16 NonLabelDoStmt -> 29: outer: do i = 1, 3
  ! CHECK:  <<DoConstruct!>> -> 29
  ! CHECK:    17 ^NonLabelDoStmt -> 28: inner: do j = 1, 5
  ! CHECK:    <<IfConstruct!>> -> 28
  ! CHECK:      18 ^IfStmt [negate] -> 28: if(j <= 1 .or. j >= 5) cycle inner
  ! CHECK:      <<IfConstruct!>> -> 28
  ! CHECK:        21 ^IfStmt [negate] -> 28: if(j == 3) goto 3
  ! CHECK:        <<IfConstruct!>> -> 27
  ! CHECK:          24 ^IfStmt -> 27: if(j == 4) cycle outer
  ! CHECK:          25 ^CycleStmt! -> 29: cycle outer
  ! CHECK:          26 EndIfStmt
  ! CHECK:        <<End IfConstruct!>>
  ! CHECK:        27 ^PrintStmt: print*, j
  ! CHECK:        23 EndIfStmt
  ! CHECK:      <<End IfConstruct!>>
  ! CHECK:      20 EndIfStmt
  ! CHECK:    <<End IfConstruct!>>
  ! CHECK:    28 ^EndDoStmt -> 17: 3 end do inner
  ! CHECK:  <<End DoConstruct!>>
  ! CHECK:  29 ^EndDoStmt -> 16: end do outer
  ! CHECK:<<End DoConstruct!>>
  outer: do i = 1, 3
    inner: do j = 1, 5
             if (j <= 1 .or. j >= 5) cycle inner
             if (j == 3) goto 3
             if (j == 4) cycle outer
             print*, j
  3        end do inner
         end do outer

  ! CHECK:   30 ^PrintStmt: print*
  print*

  ! CHECK:<<DoConstruct>> -> 40
  ! CHECK:  31 NonLabelDoStmt -> 39: do i = 1, 5
  ! CHECK:  <<IfConstruct>> -> 39
  ! CHECK:    32 ^IfStmt [negate] -> 39: if(i == 3) goto 4
  ! CHECK:    <<IfConstruct>> -> 39
  ! CHECK:      35 ^IfStmt [negate] -> 39: if(i <= 1 .or. i >= 5) cycle
  ! CHECK:      38 ^PrintStmt: print*, i
  ! CHECK:      37 EndIfStmt
  ! CHECK:    <<End IfConstruct>>
  ! CHECK:    34 EndIfStmt
  ! CHECK:  <<End IfConstruct>>
  ! CHECK:  39 EndDoStmt -> 31: 4 end do
  ! CHECK:<<End DoConstruct>>
  do i = 1, 5
     if (i == 3) goto 4
     if (i <= 1 .or. i >= 5) cycle
     print*, i
4 end do
end

! Multi-statement IF body whose last stmt is a CYCLE: the rewrite turns
! `if (cond) then; <stmts>; cycle; end if; <after>` into a structured
! `if (cond) then; <stmts>; else; <after>; end if`. The IfStmt is NOT
! negated (no `[negate]` annotation), and a synthetic ElseStmt appears
! between the THEN body and the moved <after> stmts.
subroutine block_if_cycle_multi_stmt(a)
  integer :: i, jdiag
  real :: a(:)
  jdiag = 4
  ! CHECK-LABEL: Subroutine block_if_cycle_multi_stmt
  ! CHECK: <<DoConstruct>>
  ! CHECK:   NonLabelDoStmt {{.*}}: do i = 1, 8
  ! CHECK:   <<IfConstruct>>
  ! CHECK:     {{[0-9]+}} ^IfThenStmt {{.*}}: if(i == jdiag) then
  ! CHECK-NOT:                       [negate]
  ! CHECK:     {{[0-9]+}} ^AssignmentStmt: a(i) = 0.0
  ! CHECK:     ^ElseStmt
  ! CHECK:     {{[0-9]+}} AssignmentStmt: a(i) = real(i)
  ! CHECK:     {{[0-9]+}} EndIfStmt
  ! CHECK:   <<End IfConstruct>>
  ! CHECK:   {{[0-9]+}} EndDoStmt {{.*}}: end do
  ! CHECK: <<End DoConstruct>>
  do i = 1, 8
    if (i == jdiag) then
      a(i) = 0.0
      cycle
    end if
    a(i) = real(i)
  end do
end subroutine

! IF body whose last stmt is CYCLE but contains an existing ELSE branch:
! the rewrite must NOT apply (we cannot safely splice the after-stmts past
! the existing else). The loop stays unstructured.
subroutine block_if_cycle_with_existing_else(a)
  integer :: i, jdiag
  real :: a(:)
  jdiag = 4
  ! CHECK-LABEL: Subroutine block_if_cycle_with_existing_else
  ! CHECK: <<DoConstruct!>>
  ! CHECK:   <<IfConstruct!>>
  ! CHECK:     ^IfThenStmt {{.*}}: if(i == jdiag) then
  ! CHECK:     ^AssignmentStmt: a(i) = 0.0
  ! CHECK:     ^ElseStmt: else
  ! CHECK:     AssignmentStmt: a(i) = -1.0
  ! CHECK:     CycleStmt! {{.*}}: cycle
  ! CHECK:     EndIfStmt: end if
  ! CHECK:   <<End IfConstruct!>>
  do i = 1, 8
    if (i == jdiag) then
      a(i) = 0.0
    else
      a(i) = -1.0
      cycle
    end if
    a(i) = real(i)
  end do
end subroutine
