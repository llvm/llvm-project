! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

program main
    use ISO_FORTRAN_ENV, only: compiler_options
    implicit none
    character (len = :), allocatable :: v
! CHECK: v="{{.*}}flang{{.*}} -fdebug-unparse {{.*}}"
    v = compiler_options()
    print *, v
    deallocate(v)
    close(1)
end program main
