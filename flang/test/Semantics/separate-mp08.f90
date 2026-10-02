! RUN: %python %S/test_errors.py %s %flang_fc1 -fimplicit-module-prefix -pedantic -Wno-implicit-module-prefix -Werror
! The repair still applies when its diagnostic is suppressed.
module m
  interface
    module subroutine implementation
    end subroutine implementation
  end interface
end module m

submodule(m) sm
contains
  subroutine implementation
  end subroutine implementation
end submodule sm
