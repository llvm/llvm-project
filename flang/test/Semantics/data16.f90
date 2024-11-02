! RUN: %python %S/test_errors.py %s %flang_fc1
program main
 contains
  subroutine subr
    data foo/6.66/ ! implicit declaration of "foo": ok
    integer m
    !ERROR: Implicitly typed local entity 'n' not allowed in specification expression
    real a(n)
    !ERROR: Host-associated object 'n' must not be initialized in a DATA statement
    data n/123/
    block
      real b(m)
      !ERROR: Host-associated object 'm' must not be initialized in a DATA statement
      data m/10/
    end block
  end subroutine
end program
