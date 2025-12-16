! RUN: %flang -S -emit-llvm -o - %s | FileCheck %s
! Test communication of COMPILER_OPTIONS from flang to flang -fc1.
! CHECK: [[OPTSVAR:@_QQclX[0-9a-f]+]] = {{[a-z]+}} constant [[[OPTSLEN:[0-9]+]] x i8] c"{{.*}}flang{{(\.exe)?}} {{.*}}-S -emit-llvm -o - {{.*}}compiler-options.f90"
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
