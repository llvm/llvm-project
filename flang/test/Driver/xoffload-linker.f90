! Test the -Xoffload-linker flag that forwards link commands to the clang-linker-wrapper used
! to help link offloading device libraries

! RUN:   %flang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx90a \
! RUN:      -Xoffload-linker a %s 2>&1 | FileCheck %s --check-prefix=CHECK-XLINKER

! CHECK-XLINKER: -device-linker=a{{.*}}--
