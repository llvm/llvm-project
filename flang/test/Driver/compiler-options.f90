! RUN: %flang -S -emit-llvm -o - %s | FileCheck %s
! Test communication of COMPILER_OPTIONS from flang to flang -fc1.
! In a Flang-standalone build, the driver also injects its own implicit
! search-path options (e.g. -fintrinsic-modules-path=...) ahead of the
! options given on the command line above, so allow (and ignore) an
! arbitrary prefix before the flags under test here.
! CHECK: [[OPTSVAR:@_QQclX[0-9A-Fa-f]+]] = {{[a-z]+}} constant [[[OPTSLEN:[0-9]+]] x i8] c"{{.*}}-S -emit-llvm -o -"
program main
    use ISO_FORTRAN_ENV, only: compiler_options
    implicit none
    character (len = :), allocatable :: v
! CHECK: store { ptr, i64, i32, i8, i8, i8, i8 } { ptr [[OPTSVAR]], i64 [[OPTSLEN]],
    v = compiler_options()
    print *, v
    deallocate(v)
    close(1)
end program main
