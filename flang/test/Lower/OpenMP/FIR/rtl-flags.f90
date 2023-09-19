!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEFAULT-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=45 %s -o - | FileCheck %s --check-prefix=DEFAULT-HOST-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device -fopenmp-version=45  %s -o - | FileCheck %s --check-prefix=DEFAULT-DEVICE-FIR-VERSION
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=45 %s -o - | FileCheck %s --check-prefix=DEFAULT-HOST-FIR-VERSION
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-target-debug -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DBG-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-target-debug=111 -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DBG-EQ-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-teams-oversubscription -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=TEAMS-OSUB-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-threads-oversubscription -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=THREAD-OSUB-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-no-thread-state -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=THREAD-STATE-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-no-nested-parallelism -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=NEST-PAR-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-target-debug -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism -fopenmp-assume-threads-oversubscription -fopenmp-assume-no-thread-state -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=ALL-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=DEFAULT-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-target-device -fopenmp-version=45 -o - %s | FileCheck %s --check-prefix=DEFAULT-DEVICE-FIR-VERSION
!RUN: bbc -emit-fir -fopenmp -o - %s | FileCheck %s --check-prefix=DEFAULT-HOST-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-version=45 -o - %s | FileCheck %s --check-prefix=DEFAULT-HOST-FIR-VERSION
!RUN: bbc -emit-fir -fopenmp -fopenmp-target-debug=111 -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=DBG-EQ-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-teams-oversubscription -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=TEAMS-OSUB-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-threads-oversubscription -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=THREAD-OSUB-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-no-thread-state -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=THREAD-STATE-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-no-nested-parallelism -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=NEST-PAR-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-target-debug=1 -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism -fopenmp-assume-threads-oversubscription -fopenmp-assume-no-thread-state -fopenmp-is-target-device -o - %s | FileCheck %s --check-prefix=ALL-DEVICE-FIR

!DEFAULT-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<openmp_device_version = 11>
!DEFAULT-DEVICE-FIR-SAME: omp.is_target_device = true
!DEFAULT-DEVICE-FIR-VERSION: module attributes {{{.*}}omp.flags = #omp.flags<openmp_device_version = 45>
!DEFAULT-DEVICE-FIR-VERSION-SAME: omp.is_target_device = true
!DEFAULT-DEVICE-FIR-VERSION-SAME: omp.version = #omp.version<version = 45>
!DEFAULT-HOST-FIR: module attributes {{{.*}}omp.is_target_device = false{{.*}}
!DEFAULT-HOST-FIR-VERSION: module attributes {{{.*}}omp.is_target_device = false
!DEFAULT-HOST-FIR-VERSION-SAME: omp.version = #omp.version<version = 45>
!DBG-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<debug_kind = 1, openmp_device_version = 11>
!DBG-EQ-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<debug_kind = 111, openmp_device_version = 11>
!TEAMS-OSUB-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<assume_teams_oversubscription = true, openmp_device_version = 11>
!THREAD-OSUB-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<assume_threads_oversubscription = true, openmp_device_version = 11>
!THREAD-STATE-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<assume_no_thread_state = true, openmp_device_version = 11>
!NEST-PAR-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<assume_no_nested_parallelism = true, openmp_device_version = 11>
!ALL-DEVICE-FIR: module attributes {{{.*}}omp.flags = #omp.flags<debug_kind = 1, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true, openmp_device_version = 11>
subroutine omp_subroutine()
end subroutine omp_subroutine
