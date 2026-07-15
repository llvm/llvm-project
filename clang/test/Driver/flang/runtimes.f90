! Check that Flang runtimes are passed to the linker in --driver-mode=flang.

! RUN: %clang --driver-mode=flang --rtlib=compiler-rt -### %s 2>&1 | FileCheck %s
! CHECK-DAG: clang_rt.{{[^ "]}}
! CHECK-DAG: flang_rt.{{[^ "]}}

! RUN: not %clang --driver-mode=flang -stdlib=libc++ -### %s 2>&1 | FileCheck --check-prefix=LIBCXX %s
! LIBCXX: error: unknown argument: '-stdlib=libc++'
