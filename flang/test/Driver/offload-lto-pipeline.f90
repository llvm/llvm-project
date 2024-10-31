! Test forwarding/generation of -lto-opt-pipeline to the clang-linker-wrapper
  
! RUN: %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a \
! RUN:   --target=aarch64-unknown-linux-gnu -nogpulib \
! RUN:   | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-00
! CHECK-LTO-OPT-PL-00-NOT: clang-linker-wrapper{{.*}} "--lto-opt-pipeline"

! RUN: %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a \
! RUN:   --target=aarch64-unknown-linux-gnu -nogpulib \
! RUN:   -offload-lto-opt-pipeline=lto \
! RUN:   | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-01
! CHECK-LTO-OPT-PL-01: clang-linker-wrapper{{.*}} "--lto-opt-pipeline=lto"

! RUN: %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a \
! RUN:   --target=aarch64-unknown-linux-gnu -nogpulib \
! RUN:   "-offload-lto-opt-pipeline=default<O3>" \
! RUN:   | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-02
! CHECK-LTO-OPT-PL-02: clang-linker-wrapper{{.*}} "--lto-opt-pipeline=default<O3>"

  
