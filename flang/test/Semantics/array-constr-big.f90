! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Ensure that evaluating a very large array constructor does not crash the
! compiler
program BigArray
  integer, parameter :: limit = 30
  !ERROR: Must be a constant value
  integer(foo),parameter :: jval4(limit,limit,limit) = &
    !ERROR: Must be a constant value
    reshape( (/ &
      ( &
        ( &
          (0,ii=1,limit), &
          jj=-limit,kk &
          ), &
          ( &
            i4,jj=-kk,kk &
          ), &
          ( &
            ( &
              !ERROR: Must be a constant value
              0_foo,ii=1,limit &
            ),
            jj=kk,limit &
          ), &
        kk=1,limit &
      ) /), &
             (/ limit /) )
end
