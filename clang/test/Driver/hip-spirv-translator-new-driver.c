// The --offload-new-driver was crashing when using -save-temps due to a failure in clang-linker-wrapper.
// The input and output files cannot be the same.

// RUN: %clang --offload-new-driver -### -save-temps -nogpuinc -nogpulib \
// RUN: --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv -x hip %s 2>&1 \
// RUN: | FileCheck %s

// CHECK-NOT: {{".*clang-linker-wrapper.*"}} {{.*}} "-o" "[[OUTPUT_FILE:.*.o]]" {{.*}}"[[OUTPUT_FILE]]"
// CHECK: {{".*clang-linker-wrapper.*"}} {{.*}} "-o" {{".*.hipfb"}}
