// REQUIRES: zlib

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:    --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -ggdb -gz=zlib 2>&1 | FileCheck %s

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -ggdb -gz=zlib 2>&1 | FileCheck %s

// CHECK-DAG: {{".*clang(-[0-9]+)?(.exe)?" .* "--compress-debug-sections=zlib"}}
// CHECK-DAG: {{".*lld(.exe)?" .* "--compress-debug-sections=zlib"}}
// CHECK-DAG: {{".*clang(-[0-9]+)?(.exe)?" .* "--compress-debug-sections=zlib"}}
// CHECK: "--compress-debug-sections=zlib"
