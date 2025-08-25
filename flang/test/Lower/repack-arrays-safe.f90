! Test that fir.acc_safe_temp_array_copy and fir.omp_safe_temp_array_copy
! are properly attached to fir.[un]pack_array by the lowering.

! RUN: bbc -emit-hlfir -fopenacc -frepack-arrays %s -o - | FileCheck --check-prefixes=ALL,ACC %s
! RUN: bbc -emit-hlfir -fopenmp -frepack-arrays %s -o - | FileCheck --check-prefixes=ALL,OMP %s
! RUN: bbc -emit-hlfir -fopenacc -fopenmp -frepack-arrays %s -o - | FileCheck --check-prefixes=ALL,ACCOMP %s

subroutine test(x)
  real :: x(:)
end subroutine test
! ALL-LABEL:   func.func @_QPtest(
! ACC:           %[[VAL_2:.*]] = fir.pack_array{{.*}}is_safe [#fir.acc_safe_temp_array_copy]
! ACC:           fir.unpack_array{{.*}}is_safe [#fir.acc_safe_temp_array_copy]
! OMP:           %[[VAL_2:.*]] = fir.pack_array{{.*}}is_safe [#fir.omp_safe_temp_array_copy]
! OMP:           fir.unpack_array{{.*}}is_safe [#fir.omp_safe_temp_array_copy]
! ACCOMP:           %[[VAL_2:.*]] = fir.pack_array{{.*}}is_safe [#fir.acc_safe_temp_array_copy, #fir.omp_safe_temp_array_copy]
! ACCOMP:           fir.unpack_array{{.*}}is_safe [#fir.acc_safe_temp_array_copy, #fir.omp_safe_temp_array_copy]
