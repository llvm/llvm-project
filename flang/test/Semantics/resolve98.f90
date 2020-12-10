! RUN: %S/test_errors.sh %s %t %f18

! Errors when comparing LOGICAL operands

program testCompare
  logical flag1, flag2
  if (flag1 .eqv. .false.) continue
  if (flag1 .neqv. flag2) continue
  !ERROR: LOGICAL operands must be compared using .EQV. or .NEQV.
  if (flag1 .eq. .false.) continue
  !ERROR: LOGICAL operands must be compared using .EQV. or .NEQV.
  if (flag1 .ne. flag2) continue
end program testCompare
