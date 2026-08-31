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
  !WARNING: 'implementation' is a local procedure that hides the separate module procedure interface 'm:implementation'; a call to that interface will fail to link with this local procedure. If this procedure is supposed to implement the interface, add the MODULE keyword or enable -fimplicit-module-prefix. [-Wmissing-module-prefix]
  subroutine implementation
  end subroutine implementation
end submodule sm
