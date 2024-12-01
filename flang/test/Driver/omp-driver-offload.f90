! Test that flang OpenMP and OpenMP offload related 
! commands forward or expand to the appropriate commands 
! for flang -fc1 as expected. Assumes a gfx90a, aarch64,
! and sm_70 architecture, but doesn't require one to be 
! installed or compiled for, just testing the appropriate 
! generation of jobs are created with the correct 
! corresponding arguments.

! Test regular -fopenmp with no offload
! RUN: %flang -### -fopenmp %s 2>&1 | FileCheck --check-prefixes=CHECK-OPENMP %s
! CHECK-OPENMP: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}}.f90"
! CHECK-OPENMP-NOT: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" {{.*}}.f90"

! Test regular -fopenmp with offload, and invocation filtering options
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a --offload-arch=sm_70 \
! RUN: --target=aarch64-unknown-linux-gnu -nogpulib\
! RUN:   | FileCheck %s --check-prefix=OFFLOAD-HOST-AND-DEVICE

! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a --offload-arch=sm_70 --offload-host-device \
! RUN: --target=aarch64-unknown-linux-gnu -nogpulib\
! RUN:   | FileCheck %s --check-prefix=OFFLOAD-HOST-AND-DEVICE

! OFFLOAD-HOST-AND-DEVICE: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! OFFLOAD-HOST-AND-DEVICE-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! OFFLOAD-HOST-AND-DEVICE-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "nvptx64-nvidia-cuda"
! OFFLOAD-HOST-AND-DEVICE: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"

! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a --offload-arch=sm_70 --offload-host-only \
! RUN: --target=aarch64-unknown-linux-gnu -nogpulib\
! RUN:   | FileCheck %s --check-prefix=OFFLOAD-HOST

! OFFLOAD-HOST: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! OFFLOAD-HOST-NOT: "-triple" "amdgcn-amd-amdhsa"
! OFFLOAD-HOST-NOT: "-triple" "nvptx64-nvidia-cuda"
! OFFLOAD-HOST-NOT: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"

! RUN: %flang -S -### %s 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a --offload-arch=sm_70 --offload-device-only \
! RUN: --target=aarch64-unknown-linux-gnu -nogpulib\
! RUN:   | FileCheck %s --check-prefix=OFFLOAD-DEVICE

! OFFLOAD-DEVICE: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! OFFLOAD-DEVICE-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! OFFLOAD-DEVICE-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "nvptx64-nvidia-cuda"
! OFFLOAD-DEVICE-NOT: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"

! Test regular -fopenmp with offload for basic fopenmp-is-target-device flag addition and correct fopenmp 
! RUN: %flang -### -fopenmp --offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa -nogpulib %s 2>&1 | FileCheck --check-prefixes=CHECK-OPENMP-IS-TARGET-DEVICE %s
! CHECK-OPENMP-IS-TARGET-DEVICE: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" {{.*}}.f90"

! Testing appropriate flags are gnerated and appropriately assigned by the driver when offloading
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: --target=aarch64-unknown-linux-gnu -nogpulib\
! RUN:   | FileCheck %s --check-prefix=OPENMP-OFFLOAD-ARGS
! OPENMP-OFFLOAD-ARGS: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu" {{.*}} "-fopenmp" {{.*}}.f90"
! OPENMP-OFFLOAD-ARGS-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! OPENMP-OFFLOAD-ARGS-SAME:  "-fopenmp"
! OPENMP-OFFLOAD-ARGS-SAME:  "-fopenmp-host-ir-file-path" "{{.*}}.bc" "-fopenmp-is-target-device"
! OPENMP-OFFLOAD-ARGS-SAME:  {{.*}}.f90"
! OPENMP-OFFLOAD-ARGS: "{{[^"]*}}clang-offload-packager{{.*}}" {{.*}} "--image=file={{.*}}.bc,triple=amdgcn-amd-amdhsa,arch=gfx90a,kind=openmp"
! OPENMP-OFFLOAD-ARGS-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! OPENMP-OFFLOAD-ARGS-SAME:  "-fopenmp"
! OPENMP-OFFLOAD-ARGS-SAME:  "-fembed-offload-object={{.*}}.out" {{.*}}.bc"

! Test -fopenmp with offload for RTL Flag Options
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-threads-oversubscription -nogpulib \
! RUN: | FileCheck %s --check-prefixes=CHECK-THREADS-OVS
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-assume-threads-oversubscription  \
! RUN: | FileCheck %s --check-prefixes=CHECK-THREADS-OVS
! CHECK-THREADS-OVS: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-assume-threads-oversubscription" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-teams-oversubscription  -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-TEAMS-OVS
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-assume-teams-oversubscription  \
! RUN: | FileCheck %s --check-prefixes=CHECK-TEAMS-OVS
! CHECK-TEAMS-OVS: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-assume-teams-oversubscription" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-no-nested-parallelism  -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-NEST-PAR
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-assume-no-nested-parallelism  \
! RUN: | FileCheck %s --check-prefixes=CHECK-NEST-PAR
! CHECK-NEST-PAR: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-assume-no-nested-parallelism" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-assume-no-thread-state -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-THREAD-STATE
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-assume-no-thread-state \
! RUN: | FileCheck %s --check-prefixes=CHECK-THREAD-STATE
! CHECK-THREAD-STATE: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-assume-no-thread-state" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-target-debug -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-TARGET-DEBUG
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-target-debug \
! RUN: | FileCheck %s --check-prefixes=CHECK-TARGET-DEBUG
! CHECK-TARGET-DEBUG: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-target-debug" {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-target-debug -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-TARGET-DEBUG
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-target-debug \
! RUN: | FileCheck %s --check-prefixes=CHECK-TARGET-DEBUG
! CHECK-TARGET-DEBUG-EQ: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-target-debug=111" {{.*}}.f90"

! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-target-debug -fopenmp-assume-threads-oversubscription \
! RUN: -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism \
! RUN: -fopenmp-assume-no-thread-state -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-RTL-ALL
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-target-debug -fopenmp-assume-threads-oversubscription \
! RUN: -fopenmp-assume-teams-oversubscription -fopenmp-assume-no-nested-parallelism \
! RUN: -fopenmp-assume-no-thread-state \
! RUN: | FileCheck %s --check-prefixes=CHECK-RTL-ALL
! CHECK-RTL-ALL: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" {{.*}} "-fopenmp-is-target-device" "-fopenmp-target-debug" "-fopenmp-assume-teams-oversubscription"
! CHECK-RTL-ALL: "-fopenmp-assume-threads-oversubscription" "-fopenmp-assume-no-thread-state" "-fopenmp-assume-no-nested-parallelism"
! CHECK-RTL-ALL: {{.*}}.f90"

! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN: -fopenmp-version=45 -nogpulib\
! RUN: | FileCheck %s --check-prefixes=CHECK-OPENMP-VERSION
! RUN: %flang -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=sm_70 \
! RUN: -fopenmp-targets=nvptx64-nvidia-cuda \
! RUN: -fopenmp-version=45 \
! RUN: | FileCheck %s --check-prefixes=CHECK-OPENMP-VERSION
! CHECK-OPENMP-VERSION: "{{[^"]*}}flang" "-fc1" {{.*}} "-fopenmp" "-fopenmp-version=45" {{.*}}.f90"

! Test diagnostic error when host IR file is non-existent 
! RUN: not %flang_fc1 %s -o %t 2>&1 -fopenmp -fopenmp-is-target-device \
! RUN: -fopenmp-host-ir-file-path non-existant-file.bc \
! RUN: | FileCheck %s --check-prefix=HOST-IR-MISSING
! HOST-IR-MISSING: error: provided host compiler IR file 'non-existant-file.bc' is required to generate code for OpenMP target regions but cannot be found

! RUN:   %flang -### -v --target=x86_64-unknown-linux-gnu -fopenmp  \
! RUN:      --offload-arch=gfx900 \
! RUN:      --rocm-path=%S/Inputs/rocm %s 2>&1 \
! RUN:   | FileCheck --check-prefix=ROCM-PATH %s
! ROCM-PATH: Found HIP installation: {{.*Inputs.*rocm}}, version 3.6.20214-a2917cd

! Test -fopenmp-force-usm option without offload
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-force-usm \
! RUN: --target=aarch64-unknown-linux-gnu \
! RUN:   | FileCheck %s --check-prefix=FORCE-USM-NO-OFFLOAD

! FORCE-USM-NO-OFFLOAD: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! FORCE-USM-NO-OFFLOAD-SAME: "-fopenmp" "-fopenmp-force-usm"

! Test -fopenmp-force-usm option with offload
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-force-usm --offload-arch=gfx90a \
! RUN: --target=aarch64-unknown-linux-gnu -nogpulib\
! RUN:   | FileCheck %s --check-prefix=FORCE-USM-OFFLOAD

! FORCE-USM-OFFLOAD: "{{[^"]*}}flang" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! FORCE-USM-OFFLOAD-SAME: "-fopenmp" "-fopenmp-force-usm"
! FORCE-USM-OFFLOAD-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! FORCE-USM-OFFLOAD-SAME: "-fopenmp" "-fopenmp-force-usm"

! RUN:   %flang -### -v --target=x86_64-unknown-linux-gnu -fopenmp  \
! RUN:      --offload-arch=gfx900 \
! RUN:      --rocm-path=%S/Inputs/rocm %s 2>&1 \
! RUN:   | FileCheck --check-prefix=MLINK-BUILTIN-BITCODE  %s
! MLINK-BUILTIN-BITCODE:      "{{[^"]*}}flang" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! MLINK-BUILTIN-BITCODE-SAME: "-mlink-builtin-bitcode" {{.*Inputs.*rocm.*amdgcn.*bitcode.*}}oclc_isa_version_900.bc

! Test that the -fopenmp-targets option is added to host compilation invocations
! when --offload-arch or -fopenmp-targets are set.
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp --offload-arch=gfx90a \
! RUN: --target=x86_64-unknown-linux-gnu -nogpulib \
! RUN: | FileCheck %s --check-prefix=OFFLOAD-TARGETS
! RUN: %flang -S -### %s -o %t 2>&1 \
! RUN: -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a \
! RUN: --target=x86_64-unknown-linux-gnu -nogpulib \
! RUN: | FileCheck %s --check-prefix=OFFLOAD-TARGETS

! OFFLOAD-TARGETS: "{{[^"]*}}flang" "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! OFFLOAD-TARGETS-SAME: "-fopenmp-targets=amdgcn-amd-amdhsa"
! OFFLOAD-TARGETS-NEXT: "{{[^"]*}}flang" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! OFFLOAD-TARGETS-NOT: -fopenmp-targets
! OFFLOAD-TARGETS: "{{[^"]*}}flang" "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! OFFLOAD-TARGETS-SAME: "-fopenmp-targets=amdgcn-amd-amdhsa"
