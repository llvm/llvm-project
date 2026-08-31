! REQUIRES: amdgpu-registered-target

! AMDGPU ThinLTO uses object linking in the frontend and linker wrapper.
! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp \
! RUN:   --offload-arch=gfx906 -foffload-lto=thin -nogpulib %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=THINLTO
! THINLTO: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "amdgpu9.06-amd-amdhsa"
! THINLTO-SAME: "-emit-llvm-bc"
! THINLTO-SAME: "-flto=thin"
! THINLTO-SAME: "-mllvm" "-amdgpu-enable-object-linking"
! THINLTO: "{{[^"]*}}clang-linker-wrapper"
! THINLTO-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
! THINLTO-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
! THINLTO-NOT: --device-linker=amdgpu-amd-amdhsa=-plugin-opt=-force-import-all
! THINLTO-NOT: --device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-internalize-symbols

end
