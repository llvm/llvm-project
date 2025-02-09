! Verify that the Fortran runtime libraries are present in the linker
! invocation. These libraries are added on top of other standard runtime
! libraries that the Clang driver will include.

! RUN: %flang -### --target=ppc64le-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX,UNIX-F128%f128-lib
! RUN: %flang -### --target=aarch64-apple-darwin %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,DARWIN-F128%f128-lib
! RUN: %flang -### --target=sparc-sun-solaris2.11 %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX,SOLARIS-F128%f128-lib
! RUN: %flang -### --target=x86_64-unknown-freebsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD,BSD-F128%f128-lib
! RUN: %flang -### --target=x86_64-unknown-netbsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD,BSD-F128%f128-lib
! RUN: %flang -### --target=x86_64-unknown-openbsd %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD,BSD-F128%f128-lib
! RUN: %flang -### --target=x86_64-unknown-dragonfly %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,BSD,BSD-F128%f128-lib
! RUN: %flang -### --target=x86_64-unknown-haiku %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,HAIKU,HAIKU-F128%f128-lib
! RUN: %flang -### --target=x86_64-windows-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MINGW,MINGW-F128%f128-lib
! RUN: %flang -### -rtlib=compiler-rt --target=aarch64-linux-gnu %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,UNIX,COMPILER-RT

! NOTE: Clang's driver library, clangDriver, usually adds 'oldnames' on Windows,
!       but it is not needed when compiling Fortran code and they might bring in
!       additional dependencies. Make sure its not added.
! RUN: %flang -### --target=aarch64-windows-msvc -fuse-ld= %S/Inputs/hello.f90 2>&1 | FileCheck %s --check-prefixes=CHECK,MSVC --implicit-check-not oldnames

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
! UNIX-F128NONE-NOT: lang_rt.quadmath
! SOLARIS-F128NONE-NOT: flang_rt.quadmath
! UNIX-F128LIBQUADMATH-SAME: "-lflang_rt.quadmath" "--as-needed" "-lquadmath" "--no-as-needed"
! SOLARIS-F128LIBQUADMATH-SAME: "-lflang_rt.quadmath" "-z" "ignore" "-lquadmath" "-z" "record"
! UNIX-SAME: "-lflang_rt.runtime" "-lm"
! COMPILER-RT: "{{.*}}{{\\|/}}libclang_rt.builtins.a"

! BSD-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! BSD-SAME: "[[object_file]]"
! BSD-F128NONE-NOT: flang_rt.quadmath
! BSD-F128LIBQUADMATH-SAME: "-lflang_rt.quadmath" "--as-needed" "-lquadmath" "--no-as-needed"
! BSD-SAME: -lflang_rt.runtime
! BSD-SAME: -lexecinfo

! DARWIN-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! DARWIN-SAME: "[[object_file]]"
! DARWIN-F128NONE-NOT: libflang_rt.quadmath
! DARWIN-F128LIBQUADMATH-SAME: "-lflang_rt.quadmath" "--as-needed" "-lquadmath" "--no-as-needed"
! DARWIN-SAME: -lflang_rt.runtime

! HAIKU-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! HAIKU-SAME: "[[object_file]]"
! HAIKU-F128NONE-NOT: libflang_rt.quadmath
! HAIKU-F128LIBQUADMATH-SAME: "-lflang_rt.quadmath" "--as-needed" "-lquadmath" "--no-as-needed"
! HAIKU-SAME: "-lflang_rt.runtime"

! MINGW-LABEL:  "{{.*}}ld{{(\.exe)?}}"
! MINGW-SAME: "[[object_file]]"
! MINGW-F128NONE-NOT: libflang_rt.quadmath
! MINGW-F128LIBQUADMATH-SAME: "-lflang_rt.quadmath" "--as-needed" "-lquadmath" "--no-as-needed"
! MINGW-SAME: -lflang_rt.runtime

! NOTE: This also matches lld-link (when CLANG_DEFAULT_LINKER=lld) and
!       any .exe suffix that is added when resolving to the full path of
!       (lld-)link.exe on Windows platforms. The suffix may not be added
!       when the executable is not found or on non-Windows platforms.
! MSVC-LABEL: link
! MSVC-SAME: /subsystem:console
! MSVC-SAME: "[[object_file]]"

! COMPILER-RT-NOT: "-lgcc"
! COMPILER-RT-NOT: "-lgcc_s"
