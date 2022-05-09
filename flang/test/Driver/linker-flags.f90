! Verify that the Fortran runtime libraries are present in the linker
! invocation. These libraries are added on top of other standard runtime
! libraries that the Clang driver will include.

! NOTE: The additional linker flags tested here are currently specified in
! clang/lib/Driver/Toolchains/Gnu.cpp. This makes the current implementation GNU
! (Linux) specific. The following line will make sure that this test is skipped
! on Windows. Ideally we should find a more robust way of testing this.
! REQUIRES: shell
! UNSUPPORTED: darwin, macos, system-windows

!------------
! RUN COMMAND
!------------
! Use `--ld-path` so that the linker location (used in the LABEL below) is deterministic.
! RUN: %flang -### -flang-experimental-exec --ld-path=/usr/bin/ld %S/Inputs/hello.f90 2>&1 | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! Compiler invocation to generate the object file
! CHECK-LABEL: {{.*}} "-emit-obj"
! CHECK-SAME:  "-o" "[[object_file:.*]]" {{.*}}Inputs/hello.f90

! Linker invocation to generate the executable
! CHECK-LABEL:  "/usr/bin/ld"
! CHECK-SAME: "[[object_file]]"
! CHECK-SAME: -lFortran_main
! CHECK-SAME: -lFortranRuntime
! CHECK-SAME: -lFortranDecimal
! CHECK-SAME: -lm
