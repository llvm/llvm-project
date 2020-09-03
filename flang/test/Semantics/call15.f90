! RUN: %S/test_errors.sh %s %t %f18
! C711 An assumed-type actual argument that corresponds to an assumed-rank 
! dummy argument shall be assumed-shape or assumed-rank.
subroutine s(arg1, arg2, arg3)
  type(*), dimension(..) :: arg1 ! assumed rank
  type(*), dimension(:) :: arg2 ! assumed shape
  type(*) :: arg3

  call inner(arg1) ! OK, assumed rank
  call inner(arg2) ! OK, assumed shape
  !ERROR: Assumed-type 'arg3' must be either assumed shape or assumed rank to be associated with assumed-type dummy argument 'dummy='
  call inner(arg3)

    contains
      subroutine inner(dummy)
        type(*), dimension(..) :: dummy
      end subroutine inner
end subroutine s
