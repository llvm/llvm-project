! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 -fopenmp-targets=amdgcn-amd-amdhsa -o - %s | FileCheck %s
program missing_inode
#line "/this_path_should_not_exist_on_any_system_out_there/and_if_it_does_it_will_break_the/tes.f90" 700
    ! CHECK:  define internal void @__omp_offloading_{{[0-9a-f]+}}_{{[0-9a-f]+}}__QQmain_l701
    !$omp target
        call some_routine()
    !$omp end target
end program missing_inode