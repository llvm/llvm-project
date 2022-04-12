// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx803 -nogpulib \
// RUN:   -x hip-cpp-output %s 2>&1 | FileCheck %s

// CHECK: {{".*clang-offload-bundler.*"}} {{.*}}"-output=[[HOST_PP:.*cui]]" "-output=[[DEV_PP:.*cui]]" "-unbundle"
// CHECK: {{".*clang.*"}} "-cc1" {{.*}}"-target-cpu" "gfx803" {{.*}}"-o" "[[DEV_O:.*o]]" {{.*}}"[[DEV_PP]]"
// CHECK: {{".*lld.*"}} {{.*}}"-o" "[[DEV_ISA:.*]]" "[[DEV_O]]"
// CHECK: {{".*clang-offload-bundler.*"}} {{.*}}"-input={{.*}}" "-input=[[DEV_ISA]]" "-output=[[FATBIN:.*]]"
// CHECK: {{".*clang.*"}} {{.*}}"-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fcuda-include-gpubinary" "[[FATBIN]]" {{.*}}"-o" "[[HOST_O:.*o]]" {{.*}}"[[HOST_PP]]"
// CHECK: {{".*ld.*"}} {{.*}}"[[HOST_O]]"

// RUN: %clang -### -target x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx803 -nogpulib -fgpu-rdc \
// RUN:   -x hip-cpp-output %s 2>&1 | FileCheck -check-prefix=RDC %s

// RDC: {{".*clang-offload-bundler.*"}} {{.*}}"-output=[[HOST_PP:.*cui]]" "-output=[[DEV_PP:.*cui]]" "-unbundle"
// RDC: {{".*clang.*"}} {{.*}}"-triple" "x86_64-unknown-linux-gnu"{{.*}} "-o" "[[HOST_O:.*o]]" {{.*}}"[[HOST_PP]]"
// RDC: {{".*clang-offload-bundler.*"}} {{.*}}"-output=[[HOST_PP:.*cui]]" "-output=[[DEV_PP:.*cui]]" "-unbundle"
// RDC: {{".*clang.*"}} "-cc1" {{.*}}"-target-cpu" "gfx803" {{.*}}"-o" "[[DEV_BC:.*bc]]" {{.*}}"[[DEV_PP]]"
// RDC: {{".*lld.*"}} {{.*}}"-o" "[[DEV_ISA:.*]]" "[[DEV_BC]]"
// RDC: {{".*clang-offload-bundler.*"}} {{.*}}"-input={{.*}}" "-input=[[DEV_ISA]]" "-output=[[FATBIN:.*]]"
// RDC: {{".*llvm-mc.*"}} "-o" "[[FATBIN_O:.*o]]"
// RDC: {{".*ld.*"}} {{.*}}"[[HOST_O]]" "[[FATBIN_O]]"
