! RUN: %python %S/test_errors.py %s %flang_fc1 -fimplicit-module-prefix -Wimplicit-module-prefix -Werror
! The extension warning may be requested without enabling all pedantic warnings.
module alpha
  interface
    module subroutine implementation
    end subroutine implementation
  end interface
end module alpha

submodule(alpha) beta
end submodule beta

submodule(alpha:beta) gamma
contains
  !PORTABILITY: Assuming a missing MODULE prefix on 'implementation' to repair the separate module procedure interface 'alpha:implementation' [-Wimplicit-module-prefix]
  subroutine implementation
  end subroutine implementation
end submodule gamma
