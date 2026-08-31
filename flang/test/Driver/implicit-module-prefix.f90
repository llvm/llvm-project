! REQUIRES: native
! RUN: %flang -static-libflangrt -fimplicit-module-prefix %s -o %t-enabled
! RUN: %t-enabled | FileCheck %s --check-prefix=REPAIRED
! RUN: %flang -static-libflangrt -fno-implicit-module-prefix -fimplicit-module-prefix %s -o %t-reenabled
! RUN: %t-reenabled | FileCheck %s --check-prefix=REPAIRED
! RUN: %flang -c %s -o %t-default.o
! RUN: not %flang %t-default.o -o %t-default
! RUN: %flang -Wno-missing-module-prefix -c %s -o %t-suppressed.o
! RUN: not %flang %t-suppressed.o -o %t-suppressed
! RUN: %flang -Wimplicit-module-prefix -c %s -o %t-warning-only.o
! RUN: not %flang %t-warning-only.o -o %t-warning-only
! RUN: %flang -pedantic -c %s -o %t-pedantic.o
! RUN: not %flang %t-pedantic.o -o %t-pedantic
! RUN: %flang -fimplicit-module-prefix -fno-implicit-module-prefix -c %s -o %t-disabled.o
! RUN: not %flang %t-disabled.o -o %t-disabled

module alpha
  interface
    module integer function second()
    end function second
    module integer function third()
    end function third
  end interface
end module alpha

submodule(alpha) beta
contains
  integer function second()
    second = 2
  end function second
end submodule beta

submodule(alpha:beta) gamma
contains
  integer function third()
    third = 3
  end function third
end submodule gamma

program main
  use alpha
  print *, second(), third()
end program main

! REPAIRED: 2 3
