!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 -fopenmp-targets=amdgcn-amd-amdhsa -o - %s | FileCheck %s
program missing_inode
#line "/this_path_should_not_exist_on_any_system_out_there/and_if_it_does_it_will_break_the/tes.f90" 700
    ! CHECK:  define internal void @__omp_offloading_{{[0-9a-f]+}}_{{[0-9a-f]+}}__QQmain_l701
    !$omp target
        call some_routine()
    !$omp end target
end program missing_inode
