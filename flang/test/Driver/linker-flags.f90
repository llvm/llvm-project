! Verify that the Fortran runtime libraries are present in the linker
! invocation. These libraries are added on top of other standard runtime
! libraries that the Clang driver will include.

! RUN: %flang -### --target=ppc64le-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=sparc-sun-solaris2.11 %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX
! RUN: %flang -### --target=x86_64-unknown-freebsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD
! RUN: %flang -### --target=x86_64-unknown-netbsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD
! RUN: %flang -### --target=x86_64-unknown-openbsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD
! RUN: %flang -### --target=x86_64-unknown-dragonfly %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD
! RUN: %flang -### --target=aarch64-apple-darwin %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN
! RUN: %flang -### --target=x86_64-unknown-haiku %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,HAIKU
! RUN: %flang -### --target=x86_64-windows-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MINGW

! RUN: %flang -### -rtlib=compiler-rt --target=aarch64-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s \
! RUN:   --check-prefixes=CHECK,UNIX,COMPILER-RT,%if default-unwindlib=libgcc %{DEFAULT-UNWIND-LIBGCC%} %else %{%if default-unwindlib=libunwind %{DEFAULT-UNWIND-LIBUNWIND%} %else %{NO-DEFAULT-UNWIND%}%}

! NOTE: Clang's driver library, clangDriver, usually adds 'oldnames' on Windows,
!       but it is not needed when compiling Fortran code and they might bring in
!       additional dependencies. Make sure its not added.
! RUN: %flang -### --target=aarch64-windows-msvc -fuse-ld= %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC --implicit-check-not oldnames

! RUN: %flang -### --target=ppc64le-linux-gnu -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=UNIX-STATIC-FLANGRT
! RUN: %flang -### --target=sparc-sun-solaris2.11 -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=UNIX-STATIC-FLANGRT
! RUN: %flang -### --target=x86_64-unknown-freebsd -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=BSD-STATIC-FLANGRT
! RUN: %flang -### --target=x86_64-unknown-netbsd -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=BSD-STATIC-FLANGRT
! RUN: %flang -### --target=x86_64-unknown-openbsd -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=BSD-STATIC-FLANGRT
! RUN: %flang -### --target=x86_64-unknown-dragonfly -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD-STATIC-FLANGRT
! RUN: %flang -### --target=aarch64-apple-darwin -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=DARWIN-STATIC-FLANGRT
! RUN: %flang -### --target=x86_64-unknown-haiku -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=HAIKU-STATIC-FLANGRT
! RUN: %flang -### --target=x86_64-windows-gnu -static-libflangrt %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=MINGW-STATIC-FLANGRT

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
! UNIX-SAME: "-lflang_rt.quadmath"
! UNIX-NOT: "-lquadmath"
! UNIX-SAME: "-lflang_rt.runtime" "-lm"
! COMPILER-RT: "{{.*}}{{\\|/}}libclang_rt.builtins.a"
! UNIX-STATIC-FLANGRT:   "{{.*}}{{\\|/}}libflang_rt.runtime.a"

! BSD-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! BSD-SAME: "[[object_file]]"
! BSD-SAME: "-lflang_rt.quadmath"
! BSD-NOT: "-lquadmath"
! BSD-SAME: -lflang_rt.runtime
! BSD-SAME: -lexecinfo
! BSD-STATIC-FLANGRT: "{{.*}}{{\\|/}}libflang_rt.runtime.a"

! DARWIN-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! DARWIN-SAME: "[[object_file]]"
! DARWIN-SAME: "-lflang_rt.quadmath"
! DARWIN-NOT: "-lquadmath"
! DARWIN-SAME: -lflang_rt.runtime
! DARWIN-STATIC-FLANGRT: "{{.*}}{{\\|/}}libclang_rt.runtime_osx.a"

! HAIKU-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! HAIKU-SAME: "[[object_file]]"
! HAIKU-SAME: "-lflang_rt.quadmath"
! HAIKU-NOT: "-lquadmath"
! HAIKU-SAME: "-lflang_rt.runtime"
! HAIKU-STATIC-FLANGRT: "{{.*}}{{\\|/}}libflang_rt.runtime.a"

! MINGW-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! MINGW-SAME: "[[object_file]]"
! MINGW-SAME: "-lflang_rt.quadmath"
! MINGW-NOT: "-lquadmath"
! MINGW-SAME: -lflang_rt.runtime
! MINGW-STATIC-FLANGRT: "{{.*}}{{\\|/}}libflang_rt.runtime.a"

! NOTE: This also matches lld-link (when FLANG_DEFAULT_LINKER=lld) and
!       any .exe suffix that is added when resolving to the full path of
!       (lld-)link.exe on Windows platforms. The suffix may not be added
!       when the executable is not found or on non-Windows platforms.
! MSVC-LABEL: link
! MSVC-SAME: /subsystem:console
! MSVC-SAME: "[[object_file]]"

! COMPILER-RT-NOT: "-lgcc"
! DEFAULT-UNWIND-LIBGCC: "-lgcc_s"
! DEFAULT-UNWIND-LIBUNWIND: "-lunwind"
! NO-DEFAULT-UNWIND-NOT: "-lgcc_s"
! NO-DEFAULT-UNWIND-NOT: "-lunwind"
