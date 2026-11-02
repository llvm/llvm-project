! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! Pedantic mode widens diagnostics but does not enable the repair extension.
module m
  interface
    module subroutine implementation
    end subroutine implementation
  end interface
end module m

submodule(m) sm
contains
  !PORTABILITY: Subprogram 'implementation' in this submodule is missing the MODULE prefix to implement the module procedure interface from its parent; did you mean 'MODULE SUBROUTINE'? [-Wportability]
  subroutine implementation
  end subroutine implementation
end submodule sm
