! Verify that Driver flags are known to the frontend and appropriate linker
! flags are passed to the linker.

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

! Verify that Driver flags are known to the frontend and appropriate linker
! flags are not passed to the linker.

! RUN: %flang -### --target=x86_64-unknown-freebsd -nostdlib %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NOSTDLIB %s
! RUN: %flang -### --target=x86_64-unknown-netbsd -nostdlib %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NOSTDLIB %s
! RUN: %flang -### --target=i386-pc-solaris2.11 -nostdlib %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NOSTDLIB %s

! NOSTDLIB: "{{.*}}ld{{(.exe)?}}"
! NOSTDLIB-NOT: crt{{[^.]+}}.o
! NOSTDLIB-NOT: "-lFortran_main" "-lFortranRuntime" "-lFortranDecimal" "-lm"

! RUN: %flang -### --target=x86_64-unknown-freebsd -nodefaultlibs %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NODEFAULTLIBS %s
! RUN: %flang -### --target=x86_64-unknown-netbsd -nodefaultlibs %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NODEFAULTLIBS %s
! RUN: %flang -### --target=i386-pc-solaris2.11 -nodefaultlibs %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NODEFAULTLIBS %s

! NODEFAULTLIBS: "{{.*}}ld{{(.exe)?}}"
! NODEFAULTLIBS-NOT: "-lFortran_main" "-lFortranRuntime" "-lFortranDecimal" "-lm"

! RUN: %flang -### --target=x86_64-unknown-freebsd -nostartfiles %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NOSTARTFILES %s
! RUN: %flang -### --target=x86_64-unknown-netbsd -nostartfiles %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NOSTARTFILES %s
! RUN: %flang -### --target=i386-pc-solaris2.11 -nostartfiles %s 2>&1 | FileCheck \
! RUN:     --check-prefixes=NOSTARTFILES %s

! NOSTARTFILES: "{{.*}}ld{{(.exe)?}}"
! NOSTARTFILES-NOT: crt{{[^.]+}}.o
! NOSTARTFILES: "-lFortran_main" "-lFortranRuntime" "-lFortranDecimal" "-lm"
