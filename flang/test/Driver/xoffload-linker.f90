! Test the -Xoffload-linker flag that forwards link commands to the clang-linker-wrapper used
! to help link offloading device libraries

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx90a -Xoffload-linker a %s 2>&1 | FileCheck --check-prefixes=CHECK-XLINKER %s

! CHECK-XLINKER: -device-linker=a{{.*}}-

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx90a -Xoffload-linker a -Xoffload-linker-amdgcn-amd-amdhsa b %s 2>&1 | FileCheck --check-prefixes=CHECK-XLINKER-AMDGCN %s

! CHECK-XLINKER-AMDGCN: -device-linker=a{{.*}}-device-linker=amdgcn-amd-amdhsa=b{{.*}}--
