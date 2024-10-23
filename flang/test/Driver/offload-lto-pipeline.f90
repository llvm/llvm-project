! Test forwarding/generation of -lto-opt-pipeline to the clang-linker-wrapper
  
! RUN: CLANG_USE_LINKER_WRAPPER=1 %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-00
! CHECK-LTO-OPT-PL-00-NOT: clang-linker-wrapper{{.*}} "--lto-opt-pipeline"

! RUN: CLANG_USE_LINKER_WRAPPER=1 %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib -offload-lto-opt-pipeline=lto | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-01
! RUN: CLANG_USE_LINKER_WRAPPER=1 %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib -flto | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-01
! CHECK-LTO-OPT-PL-01: clang-linker-wrapper{{.*}} "--lto-opt-pipeline=lto"

! RUN: CLANG_USE_LINKER_WRAPPER=1 %flang -### %s -o %t 2>&1 -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib "-offload-lto-opt-pipeline=default<O3>" | FileCheck %s --check-prefix=CHECK-LTO-OPT-PL-02
! CHECK-LTO-OPT-PL-02: clang-linker-wrapper{{.*}} "--lto-opt-pipeline=default<O3>"

  
