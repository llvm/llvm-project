// REQUIRES: zlib

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:    --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -ggdb -gz=zlib 2>&1 | FileCheck %s --check-prefix=NORDC

// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   -Xoffload-linker --compress-debug-sections=zlib \
// RUN:   -fgpu-rdc --offload-arch=gfx906 %s -nogpulib -nogpuinc \
// RUN:   -ggdb -gz=zlib 2>&1 | FileCheck %s --check-prefix=RDC

// NORDC-DAG: {{".*clang.*" .* "--compress-debug-sections=zlib"}}
// NORDC-DAG: {{".*lld" .* "--compress-debug-sections=zlib"}}
// NORDC-DAG: {{".*clang.*" .* "--compress-debug-sections=zlib"}}

// RDC-DAG: {{".*clang.*" .* "--compress-debug-sections=zlib"}}
// RDC-DAG: {{".*clang-linker-wrapper" .* "--device-linker=--compress-debug-sections=zlib"}}
