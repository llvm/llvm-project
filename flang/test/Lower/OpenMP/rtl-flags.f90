!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=DEFAULT-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefix=DEFAULT-HOST-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-target-debug -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=DBG-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-target-debug=111 -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=DBG-EQ-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-teams-oversubscription -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=TEAMS-OSUB-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-threads-oversubscription -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=THREAD-OSUB-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-no-thread-state -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=THREAD-STATE-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-assume-no-nested-parallelism -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=NEST-PAR-DEVICE-FIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-target-debug -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism -fopenmp-assume-threads-oversubscription -fopenmp-assume-no-thread-state -fopenmp-is-device %s -o - | FileCheck %s --check-prefix=ALL-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=DEFAULT-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -o - %s | FileCheck %s --check-prefix=DEFAULT-HOST-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-target-debug=111 -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=DBG-EQ-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-teams-oversubscription -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=TEAMS-OSUB-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-threads-oversubscription -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=THREAD-OSUB-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-no-thread-state -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=THREAD-STATE-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-assume-no-nested-parallelism -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=NEST-PAR-DEVICE-FIR
!RUN: bbc -emit-fir -fopenmp -fopenmp-target-debug=1 -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism -fopenmp-assume-threads-oversubscription -fopenmp-assume-no-thread-state -fopenmp-is-device -o - %s | FileCheck %s --check-prefix=ALL-DEVICE-FIR

!DEFAULT-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<>, omp.is_device = #omp.isdevice<is_device = true>{{.*}}}
!DEFAULT-HOST-FIR: module attributes {{{.*}},  omp.is_device = #omp.isdevice<is_device = false>{{.*}}}
!DBG-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<debug_kind = 1>{{.*}}}
!DBG-EQ-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<debug_kind = 111>{{.*}}}
!TEAMS-OSUB-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<assume_teams_oversubscription = true>{{.*}}}
!THREAD-OSUB-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<assume_threads_oversubscription = true>{{.*}}}
!THREAD-STATE-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<assume_no_thread_state = true>{{.*}}}
!NEST-PAR-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<assume_no_nested_parallelism = true>{{.*}}}
!ALL-DEVICE-FIR: module attributes {{{.*}}, omp.flags = #omp.flags<debug_kind = 1, assume_teams_oversubscription = true, assume_threads_oversubscription = true, assume_no_thread_state = true, assume_no_nested_parallelism = true>{{.*}}}
subroutine omp_subroutine()
end subroutine omp_subroutine
