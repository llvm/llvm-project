! RUN: %python %S/test_errors.py %s %flang_fc1 -Wno-portability
! The default diagnostic may be suppressed without enabling the extension.
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
