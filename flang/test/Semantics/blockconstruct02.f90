! RUN: %python %S/test_errors.py %s %flang_fc1
! C1108  --  Save statement in a BLOCK construct shall not conatin a
!            saved-entity-list that does not specify a common-block-name

program  main
  integer x, y, z
  real r, s, t
  common /argmnt2/ r, s, t
  !ERROR: 'argmnt1' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /argmnt1/
  block
    !ERROR: SAVE statement in BLOCK construct may not contain a common block name 'argmnt2'
    save /argmnt2/
  end block
end program
