! RUN %flang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx90a -Xoffload-linker a %s 2>&1 | FileCheck %s --check-prefixes=CHECK-XLINKER

! CHECK-XLINKER {{.*}}--device-linker=a{{.*}}

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp --offload-arch=gfx90a -Xoffload-linker a -Xoffload-linker-amdgcn-amd-amdhsa b %s 2>&1 | FileCheck %s --check-prefixes=CHECK-XLINKER-AMDGCN

! CHECK-XLINKER-AMDGCN: {{.*}}"--device-linker=a"{{.*}}"--device-linker=amdgcn-amd-amdhsa=b"{{.*}}

end program
