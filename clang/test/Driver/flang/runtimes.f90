! Check that Flang runtimes are passed to the linker in --driver-mode=flang.

! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=x86_64-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=i686-linux-musl -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=x86_64-pc-windows-msvc -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=x86_64-windows-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=aarch64-linux-none -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=aarch64-windows-msvc -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=ppc64le-linux-gnu -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=powerpc64-ibm-aix -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --rtlib=compiler-rt --target=sparc-sun-solaris2.11 -### %s 2>&1 | FileCheck %s --check-prefixes=FLANG-RT,CLANG-RT
! RUN: %clang --driver-mode=flang --target=arm64-apple-darwin -### %s 2>&1 | FileCheck %s --check-prefix=FLANG-RT
! RUN: %clang --driver-mode=flang --target=amd64-pc-freebsd -### %s 2>&1 | FileCheck %s --check-prefix=FLANG-RT
! RUN: %clang --driver-mode=flang --target=x86_64-unknown-haiku -### %s 2>&1 | FileCheck %s --check-prefix=FLANG-RT
! CLANG-RT-DAG: clang_rt.{{[^ "]}}
! FLANG-RT-DAG: flang_rt.{{[^ "]}}

! RUN: not %clang --driver-mode=flang -stdlib=libc++ -### %s 2>&1 | FileCheck --check-prefix=LIBCXX %s
! LIBCXX: error: unknown argument: '-stdlib=libc++'
