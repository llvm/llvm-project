! Test that -funique-internal-linkage-names / -fno-unique-internal-linkage-names are forwarded to flang -fc1.

! RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang -### -funique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=ENABLED
! RUN: %flang -### -fno-unique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=DISABLED
! RUN: %flang -### -funique-internal-linkage-names -fno-unique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=DISABLED
! RUN: %flang -### -fno-unique-internal-linkage-names -funique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=ENABLED

! DEFAULT-NOT: "-funique-internal-linkage-names"
! DEFAULT-NOT: "-fno-unique-internal-linkage-names"

! ENABLED: "-fc1"{{.*}}"-funique-internal-linkage-names"
! DISABLED-NOT: "-funique-internal-linkage-names"

subroutine host()
  call inner()
contains
  subroutine inner()
  end subroutine
end subroutine
