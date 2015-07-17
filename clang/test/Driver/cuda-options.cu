// Tests CUDA compilation pipeline construction in Driver.
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Simple compilation case:
// RUN: %clang -### -target x86_64-linux-gnu -c %s 2>&1 \
// Compile device-side to PTX assembly and make sure we use it on the host side.
// RUN:   | FileCheck -check-prefix CUDA-D1 \
// Then compile host side and incorporate device code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Typical compilation + link case:
// RUN: %clang -### -target x86_64-linux-gnu %s 2>&1 \
// Compile device-side to PTX assembly and make sure we use it on the host side
// RUN:   | FileCheck -check-prefix CUDA-D1 \
// Then compile host side and incorporate device code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Then link things.
// RUN:   -check-prefix CUDA-L %s

// Verify that -cuda-no-device disables device-side compilation and linking
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only %s 2>&1 \
// Make sure we didn't run device-side compilation.
// RUN:   | FileCheck -check-prefix CUDA-ND \
// Then compile host side and make sure we don't attempt to incorporate GPU code.
// RUN:    -check-prefix CUDA-H -check-prefix CUDA-H-NI \
// Make sure we don't link anything.
// RUN:    -check-prefix CUDA-NL %s

// Verify that -cuda-no-host disables host-side compilation and linking
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only %s 2>&1 \
// Compile device-side to PTX assembly
// RUN:   | FileCheck -check-prefix CUDA-D1 \
// Make sure there are no host cmpilation or linking.
// RUN:   -check-prefix CUDA-NH -check-prefix CUDA-NL %s

// Verify that with -S we compile host and device sides to assembly
// and incorporate device code on the host side.
// RUN: %clang -### -target x86_64-linux-gnu -S -c %s 2>&1 \
// Compile device-side to PTX assembly
// RUN:   | FileCheck -check-prefix CUDA-D1 \
// Then compile host side and incorporate GPU code.
// RUN:  -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Make sure we don't link anything.
// RUN:  -check-prefix CUDA-NL %s

// Verify that --cuda-gpu-arch option passes correct GPU
// archtecture info to device compilation.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 -c %s 2>&1 \
// Compile device-side to PTX assembly.
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1-SM35 \
// Then compile host side and incorporate GPU code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Verify that there is device-side compilation per --cuda-gpu-arch args
// and that all results are included on the host side.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 -c %s 2>&1 \
// Compile both device-sides to PTX assembly
// RUN:   | FileCheck \
// RUN: -check-prefix CUDA-D1 -check-prefix CUDA-D1-SM35 \
// RUN: -check-prefix CUDA-D2 -check-prefix CUDA-D2-SM30 \
// Then compile host side and incorporate both device-side outputs
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 -check-prefix CUDA-H-I2 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Match device-side compilation
// CUDA-D1: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// CUDA-D1-SAME: "-fcuda-is-device"
// CUDA-D1-SM35-SAME: "-target-cpu" "sm_35"
// CUDA-D1-SAME: "-o" "[[GPUBINARY1:[^"]*]]"
// CUDA-D1-SAME: "-x" "cuda"

// Match anothe device-side compilation
// CUDA-D2: "-cc1" "-triple" "nvptx{{(64)?}}-nvidia-cuda"
// CUDA-D2-SAME: "-fcuda-is-device"
// CUDA-D2-SM30-SAME: "-target-cpu" "sm_30"
// CUDA-D2-SAME: "-o" "[[GPUBINARY2:[^"]*]]"
// CUDA-D2-SAME: "-x" "cuda"

// Match no device-side compilation
// CUDA-ND-NOT: "-cc1" "-triple" "nvptx{{64?}}-nvidia-cuda"
// CUDA-ND-SAME-NOT: "-fcuda-is-device"

// Match host-side compilation
// CUDA-H: "-cc1" "-triple"
// CUDA-H-SAME-NOT: "nvptx{{64?}}-nvidia-cuda"
// CUDA-H-SAME-NOT: "-fcuda-is-device"
// CUDA-H-SAME: "-o" "[[HOSTOBJ:[^"]*]]"
// CUDA-H-SAME: "-x" "cuda"
// CUDA-H-I1-SAME: "-fcuda-include-gpubinary" "[[GPUBINARY1]]"
// CUDA-H-I2-SAME: "-fcuda-include-gpubinary" "[[GPUBINARY2]]"

// Match no GPU code inclusion.
// CUDA-H-NI-NOT: "-fcuda-include-gpubinary"

// Match no CUDA compilation
// CUDA-NH-NOT: "-cc1" "-triple"
// CUDA-NH-SAME-NOT: "-x" "cuda"

// Match linker
// CUDA-L: "{{.*}}{{ld|link}}{{(.exe)?}}"
// CUDA-L-SAME: "[[HOSTOBJ]]"

// Match no linker
// CUDA-NL-NOT: "{{.*}}{{ld|link}}{{(.exe)?}}"
