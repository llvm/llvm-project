// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// -fgpu-rdc link with output
// RUN: touch %T/obj1.o
// RUN: touch %T/obj2.o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -o executable -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --offload-arch=gfx906  %T/obj1.o %T/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,OUT %s

// -fgpu-rdc link without output
// RUN: touch %T/obj1.o
// RUN: touch %T/obj2.o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --offload-arch=gfx906  %T/obj1.o %T/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NOUT %s

// -fgpu-rdc link with output and --emit-static-lib
// RUN: touch %T/obj1.o
// RUN: touch %T/obj2.o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -o libTest.a -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --emit-static-lib \
// RUN:   --offload-arch=gfx906  %T/obj1.o %T/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,SLO %s

// -fgpu-rdc link without output and --emit-static-lib
// RUN: touch %T/obj1.o
// RUN: touch %T/obj2.o
// RUN: %clang -### -target x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --emit-static-lib \
// RUN:   --offload-arch=gfx906  %T/obj1.o %T/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,SLNO %s

// CHECK: "{{.*clang-offload-bundler.*}}" {{.*}} "-outputs=obj1-host-x86_64-unknown-linux-gnu.o,obj1-hip-amdgcn-amd-amdhsa-gfx900.o,obj1-hip-amdgcn-amd-amdhsa-gfx906.o" "-unbundle"
// CHECK: "{{.*clang-offload-bundler.*}}" {{.*}} "-outputs=obj2-host-x86_64-unknown-linux-gnu.o,obj2-hip-amdgcn-amd-amdhsa-gfx900.o,obj2-hip-amdgcn-amd-amdhsa-gfx906.o" "-unbundle"
// CHECK-NOT: {{".*/llvm-link"}}
// CHECK-NOT: {{".*/opt"}}
// CHECK-NOT: {{".*/llc"}}
// CHECK: "{{.*lld.*}}" {{.*}} "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-o" "a.out-hip-amdgcn-amd-amdhsa-gfx900" "obj1-hip-amdgcn-amd-amdhsa-gfx900.o" "obj2-hip-amdgcn-amd-amdhsa-gfx900.o"
// CHECK: "{{.*lld.*}}" {{.*}} "-mllvm" "-amdgpu-internalize-symbols"
// CHECK-SAME: "-o" "a.out-hip-amdgcn-amd-amdhsa-gfx906" "obj1-hip-amdgcn-amd-amdhsa-gfx906.o" "obj2-hip-amdgcn-amd-amdhsa-gfx906.o"
// CHECK: {{".*llvm-mc.*"}} "-triple" "amdgcn-amd-amdhsa" "-o"
// CHECK-SAME: "[[OBJBUNDLE:.*.o]]" "{{.*}}.mcin" "--filetype=obj"
// OUT: "{{.*ld.*}}" {{.*}} "-o" "executable" {{.*}} "[[OBJBUNDLE]]"
// NOUT: "{{.*ld.*}}" {{.*}} "-o" "a.out" {{.*}} "[[OBJBUNDLE]]"
// SLO: "{{.*llvm-ar.*}}" "rcsD" "libTest.a" {{.*}} "[[OBJBUNDLE]]"
// SLNO: "{{.*llvm-ar.*}}" "rcsD" "a.out" {{.*}} "[[OBJBUNDLE]]"
