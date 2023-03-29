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
