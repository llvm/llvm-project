! RUN: %python %S/test_errors.py %s %flang_fc1 -fimplicit-module-prefix -pedantic -Wno-implicit-module-prefix -Werror
! The implicit-prefix diagnostic may be suppressed while the extension is enabled.
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
