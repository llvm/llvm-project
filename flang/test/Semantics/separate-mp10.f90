! RUN: %python %S/test_errors.py %s %flang_fc1 -fimplicit-module-prefix -pedantic -Werror
! Pedantic mode reports the nonstandard repair.
module m
  interface
    module subroutine implementation
    end subroutine implementation
  end interface
end module m

submodule(m) sm
contains
  !PORTABILITY: Assuming a missing MODULE prefix on 'implementation' to repair the separate module procedure interface 'm:implementation' [-Wimplicit-module-prefix]
  subroutine implementation
  end subroutine implementation
end submodule sm
