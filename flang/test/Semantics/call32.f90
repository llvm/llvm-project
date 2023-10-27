! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! Ensure that a seemingly misparsed function reference is
! not converted to an array references of the same name if
! there's an argument keyword.
real array(1)
!ERROR: 'array' is not a callable procedure
print *, array(argument=1)
end
