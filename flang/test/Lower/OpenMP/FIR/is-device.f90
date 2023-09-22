!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefix=HOST
!RUN: %flang_fc1 -emit-fir -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE-FLAG-ONLY
!RUN: bbc -fopenmp -fopenmp-is-target-device -emit-fir -o - %s | FileCheck %s --check-prefix=DEVICE
!RUN: bbc -fopenmp -emit-fir -o - %s | FileCheck %s --check-prefix=HOST
!RUN: bbc -fopenmp-is-target-device -emit-fir -o - %s | FileCheck %s --check-prefix=DEVICE-FLAG-ONLY

!DEVICE: module attributes {{{.*}}, omp.is_target_device = true{{.*}}}
!HOST: module attributes {{{.*}}, omp.is_target_device = false{{.*}}}
!DEVICE-FLAG-ONLY: module attributes {{{.*}}"
!DEVICE-FLAG-ONLY-NOT: , omp.is_target_device = {{.*}}
!DEVICE-FLAG-ONLY-SAME: }
subroutine omp_subroutine()
end subroutine omp_subroutine
