! Test that flang-new OpenMP and OpenMP offload related 
! commands forward or expand to the appropriate commands 
! for flang-new -fc1 as expected.

! Test regular -fopenmp with no offload
! RUN: %flang -### -fopenmp %s 2>&1 | FileCheck --check-prefixes=CHECK-OPENMP %s
! CHECK-OPENMP: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}}.f90"
! CHECK-OPENMP-NOT: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" {{.*}}.f90"

! Test regular -fopenmp with offload for basic fopenmp-is-device flag addition and correct fopenmp 
! RUN: %flang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa %s 2>&1 | FileCheck --check-prefixes=CHECK-OPENMP-IS-DEVICE %s
! CHECK-OPENMP-IS-DEVICE: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" {{.*}}.f90"

! Testing fembed-offload-object and fopenmp-is-device
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: --target=aarch64-unknown-linux-gnu \
! RUN:   | FileCheck %s --check-prefixes=CHECK-OPENMP-EMBED
! CHECK-OPENMP-EMBED: "{{[^"]*}}flang-new" "-fc1" "-triple" "aarch64-unknown-linux-gnu" {{.*}} "-fopenmp" {{.*}}.f90"
! CHECK-OPENMP-EMBED-NEXT: "{{[^"]*}}flang-new" "-fc1" "-triple" "amdgcn-amd-amdhsa" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" {{.*}}.f90"
! CHECK-OPENMP-EMBED: "{{[^"]*}}clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=amdgcn-amd-amdhsa,arch=gfx90a,kind=openmp"
! CHECK-OPENMP-EMBED-NEXT: "{{[^"]*}}flang-new" "-fc1" "-triple" "aarch64-unknown-linux-gnu" {{.*}} "-fopenmp" {{.*}} "-fembed-offload-object={{.*}}.out" {{.*}}.bc"

! Test -fopenmp with offload for RTL Flag Options
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-threads-oversubscription \
! RUN: | FileCheck %s --check-prefixes=CHECK-THREADS-OVS
! CHECK-THREADS-OVS: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-assume-threads-oversubscription" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-teams-oversubscription  \
! RUN: | FileCheck %s --check-prefixes=CHECK-TEAMS-OVS
! CHECK-TEAMS-OVS: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-assume-teams-oversubscription" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-no-nested-parallelism  \
! RUN: | FileCheck %s --check-prefixes=CHECK-NEST-PAR
! CHECK-NEST-PAR: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-assume-no-nested-parallelism" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-no-thread-state \
! RUN: | FileCheck %s --check-prefixes=CHECK-THREAD-STATE
! CHECK-THREAD-STATE: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-assume-no-thread-state" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-target-debug \
! RUN: | FileCheck %s --check-prefixes=CHECK-TARGET-DEBUG
! CHECK-TARGET-DEBUG: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-target-debug" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-target-debug \
! RUN: | FileCheck %s --check-prefixes=CHECK-TARGET-DEBUG
! CHECK-TARGET-DEBUG-EQ: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-target-debug=111" {{.*}}.f90"

! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-target-debug -fopenmp-assume-threads-oversubscription \
! RUN: -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism \
! RUN: -fopenmp-assume-no-thread-state \
! RUN: | FileCheck %s --check-prefixes=CHECK-RTL-ALL
! CHECK-RTL-ALL: "{{[^"]*}}flang-new" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-device" "-fopenmp-target-debug" "-fopenmp-assume-teams-oversubscription"
! CHECK-RTL-ALL: "-fopenmp-assume-threads-oversubscription" "-fopenmp-assume-no-thread-state" "-fopenmp-assume-no-nested-parallelism"
! CHECK-RTL-ALL: {{.*}}.f90"
