// Tests CUDA compilation pipeline construction in Driver.
// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Simple compilation case. Compile device-side to PTX assembly and make sure
// we use it on the host side.
// RUN: %clang -### -target x86_64-linux-gnu -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix NOLINK %s

// Typical compilation + link case.
// RUN: %clang -### -target x86_64-linux-gnu %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// Verify that --cuda-host-only disables device-side compilation, but doesn't
// disable host-side compilation/linking.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// Same test as above, but with preceeding --cuda-device-only to make sure only
// the last option has an effect.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only --cuda-host-only %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// Verify that --cuda-device-only disables host-side compilation and linking.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// Same test as above, but with preceeding --cuda-host-only to make sure only
// the last option has an effect.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// Verify that --cuda-gpu-arch option passes the correct GPU archtecture to
// device compilation.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-gpu-arch=sm_35 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix DEVICE-SM35 -check-prefix HOST \
// RUN:    -check-prefix INCLUDES-DEVICE -check-prefix NOLINK %s

// Verify that there is one device-side compilation per --cuda-gpu-arch args
// and that all results are included on the host side.
// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=sm_35 --cuda-gpu-arch=sm_30 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix DEVICE2 -check-prefix DEVICE-SM35 \
// RUN:    -check-prefix DEVICE2-SM30 -check-prefix HOST \
// RUN:    -check-prefix HOST-NOSAVE -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix NOLINK %s

// Verify that device-side results are passed to the correct tool when
// -save-temps is used.
// RUN: %clang -### -target x86_64-linux-gnu -save-temps -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-SAVE \
// RUN:    -check-prefix HOST -check-prefix HOST-SAVE -check-prefix NOLINK %s

// Verify that device-side results are passed to the correct tool when
// -fno-integrated-as is used.
// RUN: %clang -### -target x86_64-linux-gnu -fno-integrated-as -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix HOST-NOSAVE \
// RUN:    -check-prefix HOST-AS -check-prefix NOLINK %s

// Match device-side preprocessor and compiler phases with -save-temps.
// DEVICE-SAVE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-SAVE-SAME: "-aux-triple" "x86_64--linux-gnu"
// DEVICE-SAVE-SAME: "-fcuda-is-device"
// DEVICE-SAVE-SAME: "-x" "cuda"

// DEVICE-SAVE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-SAVE-SAME: "-aux-triple" "x86_64--linux-gnu"
// DEVICE-SAVE-SAME: "-fcuda-is-device"
// DEVICE-SAVE-SAME: "-x" "cuda-cpp-output"

// Match the job that produces PTX assembly.
// DEVICE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-NOSAVE-SAME: "-aux-triple" "x86_64--linux-gnu"
// DEVICE-SAME: "-fcuda-is-device"
// DEVICE-SM35-SAME: "-target-cpu" "sm_35"
// DEVICE-SAME: "-o" "[[PTXFILE:[^"]*]]"
// DEVICE-NOSAVE-SAME: "-x" "cuda"
// DEVICE-SAVE-SAME: "-x" "ir"

// Match the call to ptxas (which assembles PTX to SASS).
// DEVICE:ptxas
// DEVICE-SM35-DAG: "--gpu-name" "sm_35"
// DEVICE-DAG: "--output-file" "[[CUBINFILE:[^"]*]]"
// DEVICE-DAG: "[[PTXFILE]]"

// Match another device-side compilation.
// DEVICE2: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE2-SAME: "-aux-triple" "x86_64--linux-gnu"
// DEVICE2-SAME: "-fcuda-is-device"
// DEVICE2-SM30-SAME: "-target-cpu" "sm_30"
// DEVICE2-SAME: "-o" "[[GPUBINARY2:[^"]*]]"
// DEVICE2-SAME: "-x" "cuda"

// Match no device-side compilation.
// NODEVICE-NOT: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// NODEVICE-SAME-NOT: "-fcuda-is-device"

// INCLUDES-DEVICE:fatbinary
// INCLUDES-DEVICE-DAG: "--create" "[[FATBINARY:[^"]*]]"
// INCLUDES-DEVICE-DAG: "--image=profile=sm_{{[0-9]+}},file=[[CUBINFILE]]"
// INCLUDES-DEVICE-DAG: "--image=profile=compute_{{[0-9]+}},file=[[PTXFILE]]"

// Match host-side preprocessor job with -save-temps.
// HOST-SAVE: "-cc1" "-triple" "x86_64--linux-gnu"
// HOST-SAVE-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// HOST-SAVE-SAME-NOT: "-fcuda-is-device"
// HOST-SAVE-SAME: "-x" "cuda"

// Match host-side compilation.
// HOST: "-cc1" "-triple" "x86_64--linux-gnu"
// HOST-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// HOST-SAME-NOT: "-fcuda-is-device"
// HOST-SAME: "-o" "[[HOSTOUTPUT:[^"]*]]"
// HOST-NOSAVE-SAME: "-x" "cuda"
// HOST-SAVE-SAME: "-x" "cuda-cpp-output"
// INCLUDES-DEVICE-SAME: "-fcuda-include-gpubinary" "[[FATBINARY]]"

// Match external assembler that uses compilation output.
// HOST-AS: "-o" "{{.*}}.o" "[[HOSTOUTPUT]]"

// Match no GPU code inclusion.
// NOINCLUDES-DEVICE-NOT: "-fcuda-include-gpubinary"

// Match no host compilation.
// NOHOST-NOT: "-cc1" "-triple"
// NOHOST-SAME-NOT: "-x" "cuda"

// Match linker.
// LINK: "{{.*}}{{ld|link}}{{(.exe)?}}"
// LINK-SAME: "[[HOSTOUTPUT]]"

// Match no linker.
// NOLINK-NOT: "{{.*}}{{ld|link}}{{(.exe)?}}"
