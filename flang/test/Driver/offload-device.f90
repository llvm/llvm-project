! -foffload-device tells the frontend we are compiling for the auxiliary target,
! i.e. not the host device.

! RUN: %flang -target aarch64-linux-gnu --offload-arch=sm_80 --offload-arch=gfx90a -### %s -fopenmp 2>&1 | FileCheck %s --check-prefixes=CHECK,OPENMP
! RUN: %flang -target aarch64-linux-gnu --offload-arch=sm_80 -### -xcuda %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CUDA

! Compiled as CUDA, device-compilation is done first
! CUDA: flang{{(\.exe)?}}" "-fc1" "-triple" "nvptx64-nvidia-cuda"
! CUDA-SAME: "-foffload-device"

! Host invocation
! CHECK: flang{{(\.exe)?}}" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-NOT: -foffload-device

! Compiled as OpenMP, device-code is compiled after host-code compilation,
! once for each --offload-arch argument
! OPENMP: flang{{(\.exe)?}}" "-fc1" "-triple" "amdgcn-amd-amdhsa"
! OPENMP-SAME: "-foffload-device"
! OPENMP: flang{{(\.exe)?}}" "-fc1" "-triple" "nvptx64-nvidia-cuda"
! OPENMP-SAME: "-foffload-device"


module device_modfile
end module
