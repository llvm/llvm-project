! RUN: %flang -fsyntax-only -fimplicit-module-prefix -Wimplicit-module-prefix %s 2>&1 | FileCheck %s --check-prefix=ENABLED
! RUN: %flang -fsyntax-only -fimplicit-module-prefix -fno-implicit-module-prefix -Wimplicit-module-prefix %s 2>&1 | FileCheck %s --allow-empty --check-prefix=DISABLED
! RUN: %flang -fsyntax-only -fno-implicit-module-prefix -fimplicit-module-prefix -Wimplicit-module-prefix %s 2>&1 | FileCheck %s --check-prefix=ENABLED

! Verify that the driver forwards the extension options to the frontend and
! that the last option wins.

module m
  interface
    module subroutine implementation()
    end subroutine
  end interface
end module

submodule (m) sm
contains
  subroutine implementation()
  end subroutine
end submodule

! ENABLED: portability: Assuming a missing MODULE prefix on 'implementation' to repair the separate module procedure interface 'm:implementation' [-Wimplicit-module-prefix]
! ENABLED-NOT: Assuming a missing MODULE prefix
! DISABLED-NOT: Assuming a missing MODULE prefix
