! Check option parsing and forwarding; lowering is covered in Integration/OpenMP.
! RUN: tco --help-hidden | FileCheck %s --check-prefix=OPTION
! RUN: not %flang_fc1 -fopenmp -emit-llvm -mmlir -openmp-disable-virtual-collapse=invalid %s -o - 2>&1 | FileCheck %s --check-prefix=INVALID
! RUN: %flang -### -fopenmp --offload-arch=gfx90a --target=aarch64-unknown-linux-gnu -nogpulib -mmlir -openmp-disable-virtual-collapse=false %s -o %t 2>&1 | FileCheck %s --check-prefix=OFFLOAD

! Pass translation options through -mmlir; -mllvm also reaches offload linkers.
! OPTION: --openmp-disable-virtual-collapse
! INVALID: 'invalid' is invalid value for boolean argument! Try 0 or 1
! OFFLOAD: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! OFFLOAD-SAME: "-mmlir" "-openmp-disable-virtual-collapse=false"
! OFFLOAD: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "amdgpu9.0a-amd-amdhsa"
! OFFLOAD-SAME: "-mmlir" "-openmp-disable-virtual-collapse=false"
! OFFLOAD: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! OFFLOAD-SAME: "-mmlir" "-openmp-disable-virtual-collapse=false"
! OFFLOAD: "{{[^"]*}}clang-linker-wrapper{{[^"]*}}"
! OFFLOAD-NOT: openmp-disable-virtual-collapse
! OFFLOAD-NOT: "-mmlir"

end
