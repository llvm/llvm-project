! RUN: %S/test_errors.sh %s %t %f18
subroutine s1
  implicit none
  real(8) :: x = 2.0
  !ERROR: The associate name 'a' is already used in this associate statement
  associate(a => x, b => x+1, a => x+2)
    x = b
  end associate
  !ERROR: No explicit type declared for 'b'
  x = b
end

subroutine s2
  !ERROR: Associate name 'a' must have a type
  associate (a => z'1')
  end associate
end

subroutine s3
! Test that associated entities are not preventing to fix
! mis-parsed function references into array references
  real :: a(10)
  associate (b => a(2:10:2))
    ! Check no complains about "Use of 'b' as a procedure"
    print *, b(1) ! OK
  end associate
  associate (c => a(2:10:2))
    ! Check the function reference has been fixed to an array reference
    !ERROR: Reference to array 'c' with empty subscript list
    print *, c()
  end associate
end
