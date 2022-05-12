! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that implicit declarations work in DATA statements
! appearing in specification parts of inner procedures; they
! should not elicit diagnostics about initialization of host
! associated objects.
program main
 contains
  subroutine subr
    data foo/6.66/ ! implicit declaration of "foo": ok
    !ERROR: Implicitly typed local entity 'n' not allowed in specification expression
    real a(n)
    !ERROR: Host-associated object 'n' must not be initialized in a DATA statement
    data n/123/
  end subroutine
end program
