! Verify that the Fortran runtime libraries are present in the linker
! invocation. These libraries are added on top of other standard runtime
! libraries that the Clang driver will include.

! RUN: %flang -### --target=ppc64le-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=aarch64-apple-darwin %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN
! RUN: %flang -### --target=sparc-sun-solaris2.11 %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=x86_64-unknown-freebsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=x86_64-unknown-netbsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=x86_64-unknown-openbsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=x86_64-unknown-dragonfly %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=x86_64-unknown-haiku %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,HAIKU
! RUN: %flang -### --target=x86_64-windows-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MINGW

! NOTE: Clang's driver library, clangDriver, usually adds 'oldnames' on Windows,
!       but it is not needed when compiling Fortran code and they might bring in
!       additional dependencies. Make sure its not added.
! RUN: %flang -### --target=aarch64-windows-msvc -fuse-ld= %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC --implicit-check-not oldnames
! RUN: %flang -### --target=aarch64-windows-msvc -fms-runtime-lib=static_dbg -fuse-ld= %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC-DEBUG --implicit-check-not oldnames
! RUN: %flang -### --target=aarch64-windows-msvc -fms-runtime-lib=dll -fuse-ld= %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC-DLL --implicit-check-not oldnames
! RUN: %flang -### --target=aarch64-windows-msvc -fms-runtime-lib=dll_dbg -fuse-ld= %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC-DLL-DEBUG --implicit-check-not oldnames

! Compiler invocation to generate the object file
! CHECK-LABEL: {{.*}} "-emit-obj"
! CHECK-SAME:  "-o" "[[object_file:.*\.o]]" {{.*}}Inputs/hello.f90

! Linker invocation to generate the executable
! NOTE: Since we are cross-compiling, the host toolchain executables may
!       run on any other platform, such as Windows that use a .exe
!       suffix. Clang's driver will try to resolve the path to the ld
!       executable and may find the GNU linker from MinGW or Cygwin.
! UNIX-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! UNIX-SAME: "[[object_file]]"
! UNIX-SAME: "-lFortran_main" "-lFortranRuntime" "-lFortranDecimal" "-lm"

! DARWIN-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! DARWIN-SAME: "[[object_file]]"
! DARWIN-SAME: -lFortran_main
! DARWIN-SAME: -lFortranRuntime
! DARWIN-SAME: -lFortranDecimal

! HAIKU-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! HAIKU-SAME: "[[object_file]]"
! HAIKU-SAME: "-lFortran_main" "-lFortranRuntime" "-lFortranDecimal"

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
! MSVC-SAME: /DEFAULTLIB:clang_rt.builtins-aarch64.lib
! MSVC-SAME: /DEFAULTLIB:libcmt
! MSVC-SAME: Fortran_main.static.lib
! MSVC-SAME: FortranRuntime.static.lib
! MSVC-SAME: FortranDecimal.static.lib
! MSVC-SAME: /subsystem:console
! MSVC-SAME: "[[object_file]]"

! MSVC-DEBUG-LABEL: link
! MSVC-DEBUG-SAME: /DEFAULTLIB:clang_rt.builtins-aarch64.lib
! MSVC-DEBUG-SAME: /DEFAULTLIB:libcmtd
! MSVC-DEBUG-SAME: Fortran_main.static_dbg.lib
! MSVC-DEBUG-SAME: FortranRuntime.static_dbg.lib
! MSVC-DEBUG-SAME: FortranDecimal.static_dbg.lib
! MSVC-DEBUG-SAME: /subsystem:console
! MSVC-DEBUG-SAME: "[[object_file]]"

! MSVC-DLL-LABEL: link
! MSVC-DLL-SAME: /DEFAULTLIB:clang_rt.builtins-aarch64.lib
! MSVC-DLL-SAME: /DEFAULTLIB:msvcrt
! MSVC-DLL-SAME: Fortran_main.dynamic.lib
! MSVC-DLL-SAME: FortranRuntime.dynamic.lib
! MSVC-DLL-SAME: FortranDecimal.dynamic.lib
! MSVC-DLL-SAME: /subsystem:console
! MSVC-DLL-SAME: "[[object_file]]"

! MSVC-DLL-DEBUG-LABEL: link
! MSVC-DLL-DEBUG-SAME: /DEFAULTLIB:clang_rt.builtins-aarch64.lib
! MSVC-DLL-DEBUG-SAME: /DEFAULTLIB:msvcrtd
! MSVC-DLL-DEBUG-SAME: Fortran_main.dynamic_dbg.lib
! MSVC-DLL-DEBUG-SAME: FortranRuntime.dynamic_dbg.lib
! MSVC-DLL-DEBUG-SAME: FortranDecimal.dynamic_dbg.lib
! MSVC-DLL-DEBUG-SAME: /subsystem:console
! MSVC-DLL-DEBUG-SAME: "[[object_file]]"
