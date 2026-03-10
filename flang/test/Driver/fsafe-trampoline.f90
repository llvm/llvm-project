! Test that -fsafe-trampoline is properly forwarded from driver to
! frontend, and that -fno-safe-trampoline (default) works.

! UNSUPPORTED: system-aix

! RUN: %flang -### -fsafe-trampoline %s 2>&1 | FileCheck %s --check-prefix=ON
! RUN: %flang -### -fno-safe-trampoline %s 2>&1 | FileCheck %s --check-prefix=OFF
! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=OFF
! RUN: %flang -### -fsafe-trampoline -fno-safe-trampoline %s 2>&1 | FileCheck %s --check-prefix=OFF

! ON: "-fsafe-trampoline"
! OFF-NOT: "-fsafe-trampoline"

program dummy
end program
