! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! Without portability warnings, a local subprogram hides an ancestor interface and leaves calls to the
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
  subroutine second
  end subroutine second
end submodule gamma
