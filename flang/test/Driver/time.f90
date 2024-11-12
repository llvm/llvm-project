! TODO: For some reason, on Windows, nothing is printed to stderr which causes
! the checks to fail. It is not clear why this is, so disable this on Windows
! until the root cause can be determined.
!
! UNSUPPORTED: system-windows

! The -time option prints timing information for the various subcommands in a
! format similar to that used by gfortran. When compiling and linking, this will
! include the time to call flang-${LLVM_VERSION_MAJOR} and the linker. Since the
! name of the linker could vary across platforms, and the flang name could also
! potentially be something different, just check that whatever is printed to
! stderr looks like timing information.

! Ideally, this should be tested on various platforms, but that requires the
! the full toolchain, including a linker to be present. The initial author of
! the test only had access to Linux on x86 which is why this is only enabled
! there. More platforms ought to be added if possible.

! REQUIRES: x86_64-linux

! RUN: %flang --target=x86_64-linux -time -c -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
! RUN: %flang --target=x86_64-linux -time -S -emit-llvm -O3 -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
! RUN: %flang --target=x86_64-linux -time -S -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
! RUN: %flang --target=x86_64-linux -time -o /dev/null %s 2>&1 \
! RUN:     | FileCheck %s --check-prefix=COMPILE-AND-LINK

! COMPILE-ONLY: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}
! COMPILE-ONLY-NOT: {{.}}

! COMPILE-AND-LINK: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}
! COMPILE-AND-LINK: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}

end program
