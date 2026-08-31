! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! A local subprogram hides an ancestor interface and leaves calls to the
! ancestor's separate module procedure undefined at link time.
module alpha
  interface
    module subroutine second
    end subroutine second
  end interface
end module alpha

submodule(alpha) beta
end submodule beta

submodule(alpha:beta) gamma
contains
  !WARNING: 'second' is a local procedure that hides the separate module procedure interface 'alpha:second'; a call to that interface will fail to link with this local procedure. If this procedure is supposed to implement the interface, add the MODULE keyword or enable -fimplicit-module-prefix. [-Wmissing-module-prefix]
  subroutine second
  end subroutine second
end submodule gamma
