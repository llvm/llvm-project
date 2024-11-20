! Test the -Xoffload-linker flag that forwards link commands to the clang-linker-wrapper used
! to help link offloading device libraries

! RUN: %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib -Xoffload-linker a | FileCheck %s --check-prefix=CHECK-XLINKER

! CHECK-XLINKER: "{{[^"]*}}clang-linker-wrapper{{.*}}"{{.*}}"--device-linker=a"{{.*}}

! RUN: %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib -Xoffload-linker a -Xoffload-linker-amdgcn-amd-amdhsa b | FileCheck %s --check-prefix=CHECK-XLINKER-AMDGCN

! CHECK-XLINKER-AMDGCN: "{{[^"]*}}clang-linker-wrapper{{.*}}"{{.*}}"--device-linker=a"{{.*}}"--device-linker=amdgcn-amd-amdhsa=b"{{.*}}
