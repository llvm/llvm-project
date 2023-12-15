! Verify that rdynamic flag adds -export-dynamic flag and passes it on to the linker.

! RUN: %flang -### --target=x86_64-linux-gnu -rdynamic %s 2>&1 | FileCheck --check-prefixes=GNU-LINKER-OPTIONS %s
! RUN: %flang -### --target=aarch64-linux-none -rdynamic %s 2>&1 | FileCheck --check-prefixes=AARCH-LINKER-OPTIONS %s

! GNU-LINKER-OPTIONS: "{{.*}}ld"
! GNU-LINKER-OPTIONS-SAME: "-export-dynamic"

! AARCH-LINKER-OPTIONS: "{{.*}}ld"
! AARCH-LINKER-OPTIONS-SAME: "-export-dynamic"
