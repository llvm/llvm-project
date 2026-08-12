! RUN: %flang_fc1 -fsyntax-only %s
! dummy procedure length may depend on dummy arguments
! actual with fixed length is compatible when lengths match at call
program test
  character(4), external :: ext
  call s(ext, ext, 2)
contains
  subroutine s(fun, fun_alt, n)
    integer :: n
    character(2 * n), external :: fun
    character(n * (n + 1) - n**2 + n), external :: fun_alt
    print *, fun()
    print *, fun_alt()
  end subroutine
end program

function ext()
  character(4) :: ext
  ext = 'okko'
end function
