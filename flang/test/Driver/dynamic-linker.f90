! Verify that certain linker flags are known to the frontend and are passed on
! to the linker.

! RUN: %flang -### --target=x86_64-linux-gnu -rpath /path/to/dir -shared \
! RUN:     -static %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=GNU-LINKER-OPTIONS %s
! RUN: %flang -### --target=x86_64-windows-msvc -rpath /path/to/dir -shared \
! RUN:     -static %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=MSVC-LINKER-OPTIONS %s

! TODO: Could the linker have an extension or a suffix?
! GNU-LINKER-OPTIONS: "{{.*}}ld{{(.exe)?}}"
! GNU-LINKER-OPTIONS-SAME: "-shared"
! GNU-LINKER-OPTIONS-SAME: "-static"
! GNU-LINKER-OPTIONS-SAME: "-rpath" "/path/to/dir"

! For MSVC, adding -static does not add any additional linker options.
! MSVC-LINKER-OPTIONS: "{{.*}}link{{(.exe)?}}"
! MSVC-LINKER-OPTIONS-SAME: "-dll"
! MSVC-LINKER-OPTIONS-SAME: "-rpath" "/path/to/dir"
