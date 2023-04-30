! Verify that the Fortran runtime libraries are present in the linker
! invocation. These libraries are added on top of other standard runtime
! libraries that the Clang driver will include.

! RUN: %flang -### -flang-experimental-exec -target ppc64le-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,GNU
! RUN: %flang -### -flang-experimental-exec -target aarch64-apple-darwin %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN
! RUN: %flang -### -flang-experimental-exec -target x86_64-windows-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MINGW

! NOTE: Clang's driver library, clangDriver, usually adds 'libcmt' and
!       'oldnames' on Windows, but they are not needed when compiling
!       Fortran code and they might bring in additional dependencies.
!       Make sure they're not added.
! RUN: %flang -### -flang-experimental-exec -target aarch64-windows-msvc %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC --implicit-check-not libcmt --implicit-check-not oldnames

! Check linker invocation to generate shared object (only GNU toolchain for now)
! Output should not contain any undefined reference to _QQmain since it is not
! considered a valid entry point for shared objects, which are usually specified
! using the bind attribute.
! RUN: %flang -### -flang-experimental-exec -shared -target x86_64-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,GNU-SHARED --implicit-check-not _QQmain
! RUN: %flang -### -flang-experimental-exec -shared -target aarch64-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,GNU-SHARED --implicit-check-not _QQmain
! RUN: %flang -### -flang-experimental-exec -shared -target riscv64-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,GNU-SHARED --implicit-check-not _QQmain

! Compiler invocation to generate the object file
! CHECK-LABEL: {{.*}} "-emit-obj"
! CHECK-SAME:  "-o" "[[object_file:.*\.o]]" {{.*}}Inputs/hello.f90

! Linker invocation to generate the executable
! NOTE: Since we are cross-compiling, the host toolchain executables may
!       run on any other platform, such as Windows that use a .exe
!       suffix. Clang's driver will try to resolve the path to the ld
!       executable and may find the GNU linker from MinGW or Cygwin.
! GNU-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! GNU-SAME: "[[object_file]]"
! GNU-SAME: --undefined=_QQmain
! GNU-SAME: -lFortran_main
! GNU-SAME: -lFortranRuntime
! GNU-SAME: -lFortranDecimal
! GNU-SAME: -lm

! DARWIN-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! DARWIN-SAME: "[[object_file]]"
! DARWIN-SAME: -lFortran_main
! DARWIN-SAME: -lFortranRuntime
! DARWIN-SAME: -lFortranDecimal

! MINGW-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! MINGW-SAME: "[[object_file]]"
! MINGW-SAME: -lFortran_main
! MINGW-SAME: -lFortranRuntime
! MINGW-SAME: -lFortranDecimal

! NOTE: This also matches lld-link (when CLANG_DEFAULT_LINKER=lld) and
!       any .exe suffix that is added when resolving to the full path of
!       (lld-)link.exe on Windows platforms. The suffix may not be added
!       when the executable is not found or on non-Windows platforms.
! MSVC-LABEL: link
! MSVC-SAME: Fortran_main.lib
! MSVC-SAME: FortranRuntime.lib
! MSVC-SAME: FortranDecimal.lib
! MSVC-SAME: /subsystem:console
! MSVC-SAME: "[[object_file]]"

! Linker invocation to generate a shared object
! GNU-SHARED-LABEL:  "{{.*}}ld"
! GNU-SHARED-SAME: "[[object_file]]"
! GNU-SHARED-SAME: -lFortranRuntime
! GNU-SHARED-SAME: -lFortranDecimal
