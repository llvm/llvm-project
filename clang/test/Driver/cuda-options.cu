// Tests CUDA compilation pipeline construction in Driver.
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Simple compilation case:
// RUN: %clang -### -target x86_64-linux-gnu -c %s 2>&1 \
// Compile device-side to PTX assembly and make sure we use it on the host side.
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS\
// Then compile host side and incorporate device code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Typical compilation + link case:
// RUN: %clang -### -target x86_64-linux-gnu %s 2>&1 \
// Compile device-side to PTX assembly and make sure we use it on the host side
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS\
// Then compile host side and incorporate device code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Then link things.
// RUN:   -check-prefix CUDA-L %s

// Verify that --cuda-host-only disables device-side compilation and linking
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only %s 2>&1 \
// Make sure we didn't run device-side compilation.
// RUN:   | FileCheck -check-prefix CUDA-ND \
// Then compile host side and make sure we don't attempt to incorporate GPU code.
// RUN:    -check-prefix CUDA-H -check-prefix CUDA-H-NI \
// Linking is allowed to happen, even if we're missing GPU code.
// RUN:    -check-prefix CUDA-L %s

// Same test as above, but with preceeding --cuda-device-only to make
// sure only last option has effect.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only --cuda-host-only %s 2>&1 \
// Make sure we didn't run device-side compilation.
// RUN:   | FileCheck -check-prefix CUDA-ND \
// Then compile host side and make sure we don't attempt to incorporate GPU code.
// RUN:    -check-prefix CUDA-H -check-prefix CUDA-H-NI \
// Linking is allowed to happen, even if we're missing GPU code.
// RUN:    -check-prefix CUDA-L %s

// Verify that --cuda-device-only disables host-side compilation and linking
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only %s 2>&1 \
// Compile device-side to PTX assembly
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS\
// Make sure there are no host cmpilation or linking.
// RUN:   -check-prefix CUDA-NH -check-prefix CUDA-NL %s

// Same test as above, but with preceeding --cuda-host-only to make
// sure only last option has effect.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only --cuda-device-only %s 2>&1 \
// Compile device-side to PTX assembly
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS\
// Make sure there are no host cmpilation or linking.
// RUN:   -check-prefix CUDA-NH -check-prefix CUDA-NL %s

// Verify that with -S we compile host and device sides to assembly
// and incorporate device code on the host side.
// RUN: %clang -### -target x86_64-linux-gnu -S -c %s 2>&1 \
// Compile device-side to PTX assembly
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS\
// Then compile host side and incorporate GPU code.
// RUN:  -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Make sure we don't link anything.
// RUN:  -check-prefix CUDA-NL %s

// Verify that --cuda-gpu-arch option passes correct GPU
// archtecture info to device compilation.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 -c %s 2>&1 \
// Compile device-side to PTX assembly.
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS \
// RUN:   -check-prefix CUDA-D1-SM35 \
// Then compile host side and incorporate GPU code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-H-I1 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Verify that there is device-side compilation per --cuda-gpu-arch args
// and that all results are included on the host side.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:        --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 -c %s 2>&1 \
// Compile both device-sides to PTX assembly
// RUN:   | FileCheck \
// RUN: -check-prefix CUDA-D1 -check-prefix CUDA-D1NS -check-prefix CUDA-D1-SM35 \
// RUN: -check-prefix CUDA-D2 -check-prefix CUDA-D2-SM30 \
// Then compile host side and incorporate both device-side outputs
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-HNS \
// RUN:   -check-prefix CUDA-H-I1 -check-prefix CUDA-H-I2 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Verify that device-side results are passed to correct tool when
// -save-temps is used
// RUN: %clang -### -target x86_64-linux-gnu -save-temps -c %s 2>&1 \
// Compile device-side to PTX assembly and make sure we use it on the host side.
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1S \
// Then compile host side and incorporate device code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-HS -check-prefix CUDA-HS-I1 \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// Verify that device-side results are passed to correct tool when
// -fno-integrated-as is used
// RUN: %clang -### -target x86_64-linux-gnu -fno-integrated-as -c %s 2>&1 \
// Compile device-side to PTX assembly and make sure we use it on the host side.
// RUN:   | FileCheck -check-prefix CUDA-D1 -check-prefix CUDA-D1NS \
// Then compile host side and incorporate device code.
// RUN:   -check-prefix CUDA-H -check-prefix CUDA-HNS -check-prefix CUDA-HS-I1 \
// RUN:   -check-prefix CUDA-H-AS \
// Make sure we don't link anything.
// RUN:   -check-prefix CUDA-NL %s

// --cuda-host-only should never trigger unused arg warning.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only -c %s 2>&1 | \
// RUN:    FileCheck -check-prefix CUDA-NO-UNUSED-CHO %s
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only -x c -c %s 2>&1 | \
// RUN:    FileCheck -check-prefix CUDA-NO-UNUSED-CHO %s

// --cuda-device-only should not produce warning compiling CUDA files
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only -c %s 2>&1 | \
// RUN:    FileCheck -check-prefix CUDA-NO-UNUSED-CDO %s

// --cuda-device-only should warn during non-CUDA compilation.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only -x c -c %s 2>&1 | \
// RUN:    FileCheck -check-prefix CUDA-UNUSED-CDO %s

// Match device-side preprocessor, and compiler phases with -save-temps
// CUDA-D1S: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA-D1S-SAME: "-aux-triple" "x86_64--linux-gnu"
// CUDA-D1S-SAME: "-fcuda-is-device"
// CUDA-D1S-SAME: "-x" "cuda"

// CUDA-D1S: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA-D1S-SAME: "-aux-triple" "x86_64--linux-gnu"
// CUDA-D1S-SAME: "-fcuda-is-device"
// CUDA-D1S-SAME: "-x" "cuda-cpp-output"

// Match the job that produces PTX assembly
// CUDA-D1: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA-D1NS-SAME: "-aux-triple" "x86_64--linux-gnu"
// CUDA-D1-SAME: "-fcuda-is-device"
// CUDA-D1-SM35-SAME: "-target-cpu" "sm_35"
// CUDA-D1-SAME: "-o" "[[GPUBINARY1:[^"]*]]"
// CUDA-D1NS-SAME: "-x" "cuda"
// CUDA-D1S-SAME: "-x" "ir"

// Match another device-side compilation
// CUDA-D2: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA-D2-SAME: "-aux-triple" "x86_64--linux-gnu"
// CUDA-D2-SAME: "-fcuda-is-device"
// CUDA-D2-SM30-SAME: "-target-cpu" "sm_30"
// CUDA-D2-SAME: "-o" "[[GPUBINARY2:[^"]*]]"
// CUDA-D2-SAME: "-x" "cuda"

// Match no device-side compilation
// CUDA-ND-NOT: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA-ND-SAME-NOT: "-fcuda-is-device"

// Match host-side preprocessor job with -save-temps
// CUDA-HS: "-cc1" "-triple" "x86_64--linux-gnu"
// CUDA-HS-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// CUDA-HS-SAME-NOT: "-fcuda-is-device"
// CUDA-HS-SAME: "-x" "cuda"

// Match host-side compilation
// CUDA-H: "-cc1" "-triple" "x86_64--linux-gnu"
// CUDA-H-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// CUDA-H-SAME-NOT: "-fcuda-is-device"
// CUDA-H-SAME: "-o" "[[HOSTOUTPUT:[^"]*]]"
// CUDA-HNS-SAME: "-x" "cuda"
// CUDA-HS-SAME: "-x" "cuda-cpp-output"
// CUDA-H-I1-SAME: "-fcuda-include-gpubinary" "[[GPUBINARY1]]"
// CUDA-H-I2-SAME: "-fcuda-include-gpubinary" "[[GPUBINARY2]]"

// Match external assembler that uses compilation output
// CUDA-H-AS: "-o" "{{.*}}.o" "[[HOSTOUTPUT]]"

// Match no GPU code inclusion.
// CUDA-H-NI-NOT: "-fcuda-include-gpubinary"

// Match no CUDA compilation
// CUDA-NH-NOT: "-cc1" "-triple"
// CUDA-NH-SAME-NOT: "-x" "cuda"

// Match linker
// CUDA-L: "{{.*}}{{ld|link}}{{(.exe)?}}"
// CUDA-L-SAME: "[[HOSTOUTPUT]]"

// Match no linker
// CUDA-NL-NOT: "{{.*}}{{ld|link}}{{(.exe)?}}"

// CUDA-NO-UNUSED-CHO-NOT: warning: argument unused during compilation: '--cuda-host-only'
// CUDA-UNUSED-CDO: warning: argument unused during compilation: '--cuda-device-only'
// CUDA-NO-UNUSED-CDO-NOT: warning: argument unused during compilation: '--cuda-device-only'
