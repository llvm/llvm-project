! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

program main
    use ISO_FORTRAN_ENV, only: compiler_version
    implicit none
    character (len = :), allocatable :: v
! CHECK: v="{{.*}}flang version {{[0-9]+\.[0-9.]+.*}}"
    v = compiler_version()
    print *, v
    deallocate(v)
    close(1)
end program main
