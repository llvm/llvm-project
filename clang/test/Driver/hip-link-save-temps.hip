// -fgpu-rdc link with output
// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/obj1.o
// RUN: touch %t/obj2.o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -o executable -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --no-offload-new-driver --offload-arch=gfx906  %t/obj1.o %t/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,OUT %s

// -fgpu-rdc link without output
// RUN: touch %t/obj1.o
// RUN: touch %t/obj2.o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --no-offload-new-driver --hip-link -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --offload-arch=gfx906  %t/obj1.o %t/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,NOUT %s

// -fgpu-rdc link with output and --emit-static-lib
// RUN: touch %t/obj1.o
// RUN: touch %t/obj2.o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -o libTest.a -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --no-offload-new-driver --emit-static-lib \
// RUN:   --offload-arch=gfx906  %t/obj1.o %t/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,SLO %s

// -fgpu-rdc link without output and --emit-static-lib
// RUN: touch %t/obj1.o
// RUN: touch %t/obj2.o
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -save-temps \
// RUN:   --hip-link -fgpu-rdc --cuda-gpu-arch=gfx900 \
// RUN:   --no-offload-new-driver --emit-static-lib \
// RUN:   --offload-arch=gfx906  %t/obj1.o %t/obj2.o 2>&1 | \
// RUN:   FileCheck -check-prefixes=CHECK,SLNO %s

// CHECK: "{{.*clang-offload-bundler.*}}" {{.*}} "-output=obj1-host-x86_64-unknown-linux-gnu.o" "-output=obj1-hip-amdgcn-amd-amdhsa-gfx900.o" "-output=obj1-hip-amdgcn-amd-amdhsa-gfx906.o" "-unbundle"
// CHECK: "{{.*clang-offload-bundler.*}}" {{.*}} "-output=obj2-host-x86_64-unknown-linux-gnu.o" "-output=obj2-hip-amdgcn-amd-amdhsa-gfx900.o" "-output=obj2-hip-amdgcn-amd-amdhsa-gfx906.o" "-unbundle"
// CHECK-NOT: {{".*/llvm-link"}}
// CHECK-NOT: {{".*/opt"}}
// CHECK-NOT: {{".*/llc"}}
// CHECK: "{{.*lld.*}}" {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// CHECK-SAME: "-o" "[[HIPFB1:.+]]" "obj1-hip-amdgcn-amd-amdhsa-gfx900.o" "obj2-hip-amdgcn-amd-amdhsa-gfx900.o"
// CHECK: "{{.*lld.*}}" {{.*}} "-plugin-opt=-amdgpu-internalize-symbols"
// CHECK-SAME: "-o" "[[HIPFB2:.+]]" "obj1-hip-amdgcn-amd-amdhsa-gfx906.o" "obj2-hip-amdgcn-amd-amdhsa-gfx906.o"
// CHECK: "{{.*clang.*}}" "-target" "x86_64-unknown-linux-gnu" "-o" "[[OBJBUNDLE:.+.o]]" "-x" "assembler" "{{.*}}.mcin" "-c"
// OUT: "{{.*ld.*}}" {{.*}} "-o" "executable" {{.*}} "[[OBJBUNDLE]]"
// NOUT: "{{.*ld.*}}" {{.*}} "-o" "a.out" {{.*}} "[[OBJBUNDLE]]"
// SLO: "{{.*llvm-ar.*}}" "rcsD" "libTest.a" {{.*}} "[[OBJBUNDLE]]"
// SLNO: "{{.*llvm-ar.*}}" "rcsD" "a.out" {{.*}} "[[OBJBUNDLE]]"
