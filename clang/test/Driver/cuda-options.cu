// Tests CUDA compilation pipeline construction in Driver.

// Simple compilation case. Compile device-side to PTX assembly and make sure
// we use it on the host side.
// RUN: %clang -### --cuda-include-ptx=all -target x86_64-linux-gnu -c -nogpulib -nogpuinc %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix NOLINK %s

// Typical compilation + link case.
// RUN: %clang -### --cuda-include-ptx=all -target x86_64-linux-gnu -nogpulib -nogpuinc %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// Verify that --cuda-host-only disables device-side compilation, but doesn't
// disable host-side compilation/linking.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-host-only -nogpulib -nogpuinc %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// Verify that --cuda-device-only disables host-side compilation and linking.
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only -nogpulib -nogpuinc %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// Check that the last of --cuda-compile-host-device, --cuda-host-only, and
// --cuda-device-only wins.

// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:    --cuda-host-only -nogpulib -nogpuinc %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// RUN: %clang -### --target=x86_64-linux-gnu --cuda-compile-host-device \
// RUN:    --cuda-host-only --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefix NODEVICE -check-prefix HOST \
// RUN:    -check-prefix NOINCLUDES-DEVICE -check-prefix LINK %s

// RUN: %clang -### --target=x86_64-linux-gnu --cuda-host-only \
// RUN:    -nogpulib -nogpuinc --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// RUN: %clang -### --target=x86_64-linux-gnu --cuda-compile-host-device \
// RUN:    -nogpulib -nogpuinc --cuda-device-only %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix NOHOST -check-prefix NOLINK %s

// RUN: %clang -### --cuda-include-ptx=all --target=x86_64-linux-gnu --cuda-host-only \
// RUN:   -nogpulib -nogpuinc --cuda-compile-host-device %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// RUN: %clang -### --cuda-include-ptx=all --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-compile-host-device %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix INCLUDES-DEVICE \
// RUN:    -check-prefix LINK %s

// Verify that --cuda-gpu-arch option passes the correct GPU architecture to
// device compilation.
// RUN: %clang -### -nogpulib -nogpuinc --cuda-include-ptx=all --target=x86_64-linux-gnu --cuda-gpu-arch=sm_52 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix DEVICE-SM52 -check-prefix HOST \
// RUN:    -check-prefix INCLUDES-DEVICE -check-prefix NOLINK %s

// Verify that there is one device-side compilation per --cuda-gpu-arch args
// and that all results are included on the host side.
// RUN: %clang -### --cuda-include-ptx=all --target=x86_64-linux-gnu \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes DEVICE,DEVICE-NOSAVE,DEVICE2 \
// RUN:             -check-prefixes DEVICE-SM52,DEVICE2-SM60 \
// RUN:             -check-prefixes INCLUDES-DEVICE,INCLUDES-DEVICE2 \
// RUN:             -check-prefixes HOST,HOST-NOSAVE,NOLINK %s

// Verify that device-side results are passed to the correct tool when
// -save-temps is used.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc -save-temps -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-SAVE \
// RUN:    -check-prefix HOST -check-prefix HOST-SAVE -check-prefix NOLINK %s

// Verify that device-side results are passed to the correct tool when
// -fno-integrated-as is used.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc -fno-integrated-as -c %s 2>&1 \
// RUN: | FileCheck -check-prefix DEVICE -check-prefix DEVICE-NOSAVE \
// RUN:    -check-prefix HOST -check-prefix HOST-NOSAVE \
// RUN:    -check-prefix HOST-AS -check-prefix NOLINK %s

// Verify that --[no-]cuda-gpu-arch arguments are handled correctly.
// a) --no-cuda-gpu-arch=X negates preceding --cuda-gpu-arch=X
// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-gpu-arch=sm_70 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM52,NOARCH-SM60,NOARCH-SM70 %s

// b) --no-cuda-gpu-arch=X negates more than one preceding --cuda-gpu-arch=X
// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-gpu-arch=sm_70 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM52,NOARCH-SM60,NOARCH-SM70 %s

// c) if --no-cuda-gpu-arch=X negates all preceding --cuda-gpu-arch=X
//    we default to sm_52 -- same as if no --cuda-gpu-arch were passed.
// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_60 \
// RUN:   --no-cuda-gpu-arch=sm_70 --no-cuda-gpu-arch=sm_60 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM52,NOARCH-SM60,NOARCH-SM70 %s

// d) --no-cuda-gpu-arch=X is a no-op if there's no preceding --cuda-gpu-arch=X
// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52\
// RUN:   --no-cuda-gpu-arch=sm_70 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM52,ARCH-SM60,NOARCH-SM70 %s

// e) --no-cuda-gpu-arch=X does not affect following --cuda-gpu-arch=X
// RUN: %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --no-cuda-gpu-arch=sm_70 --no-cuda-gpu-arch=sm_52 \
// RUN:   --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_52 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes ARCH-SM52,NOARCH-SM60,ARCH-SM70 %s

// f) --no-cuda-gpu-arch=all negates all preceding --cuda-gpu-arch=X
// RUN: %clang -### -target x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-version-check --no-cuda-gpu-arch=all \
// RUN:   --cuda-gpu-arch=sm_70 \
// RUN:   -c --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes NOARCH-SM52,NOARCH-SM60,ARCH-SM70 %s

// g) There's no --cuda-gpu-arch=all
// RUN: not %clang -### --target=x86_64-linux-gnu --cuda-device-only \
// RUN:   -nogpulib -nogpuinc --cuda-gpu-arch=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefix ARCHALLERROR %s


// Verify that --[no-]cuda-include-ptx arguments are handled correctly.
// a) by default we're not including PTX for all GPUs.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --cuda-include-ptx=all --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM60,PTX-SM52 %s

// b) --no-cuda-include-ptx=all disables PTX inclusion for all GPUs
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-include-ptx=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,NOPTX-SM60,NOPTX-SM52 %s

// c) --no-cuda-include-ptx=sm_XX disables PTX inclusion for that GPU only.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-include-ptx=sm_60 --cuda-include-ptx=sm_52 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,NOPTX-SM60,PTX-SM52 %s
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-include-ptx=sm_52 --cuda-include-ptx=sm_60 \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM60,NOPTX-SM52 %s

// d) --cuda-include-ptx=all overrides preceding --no-cuda-include-ptx=all
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-include-ptx=all --cuda-include-ptx=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM60,PTX-SM52 %s

// e) --cuda-include-ptx=all overrides preceding --no-cuda-include-ptx=sm_XX
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --cuda-gpu-arch=sm_60 --cuda-gpu-arch=sm_52 \
// RUN:   --no-cuda-include-ptx=sm_52 --cuda-include-ptx=all \
// RUN:   -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes FATBIN-COMMON,PTX-SM60,PTX-SM52 %s

// Verify -flto=thin -fwhole-program-vtables handling. This should result in
// both options being passed to the host compilation, with neither passed to
// the device compilation.
// RUN: %clang -### --cuda-include-ptx=sm_60 --target=x86_64-linux-gnu -nogpulib -nogpuinc -c -flto=thin -fwhole-program-vtables %s 2>&1 \
// RUN: | FileCheck -check-prefixes DEVICE,DEVICE-NOSAVE,HOST,NOLINK,THINLTOWPD %s
// THINLTOWPD-NOT: error: invalid argument '-fwhole-program-vtables' only allowed with '-flto'

// ARCH-SM52: "-cc1"{{.*}}"-target-cpu" "sm_52"
// NOARCH-SM52-NOT: "-cc1"{{.*}}"-target-cpu" "sm_52"
// ARCH-SM60: "-cc1"{{.*}}"-target-cpu" "sm_60"
// NOARCH-SM60-NOT: "-cc1"{{.*}}"-target-cpu" "sm_60"
// ARCH-SM70: "-cc1"{{.*}}"-target-cpu" "sm_70"
// NOARCH-SM70-NOT: "-cc1"{{.*}}"-target-cpu" "sm_70"
// ARCHALLERROR: error: unsupported CUDA gpu architecture: all

// Match device-side preprocessor and compiler phases with -save-temps.
// DEVICE-SAVE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-SAVE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// DEVICE-SAVE-SAME: "-fcuda-is-device"
// DEVICE-SAVE-SAME: "-x" "cuda"

// DEVICE-SAVE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-SAVE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// DEVICE-SAVE-SAME: "-fcuda-is-device"
// DEVICE-SAVE-SAME: "-x" "cuda-cpp-output"

// Match the job that produces PTX assembly.
// DEVICE: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE-NOSAVE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// THINLTOWPD-NOT: "-flto=thin"
// DEVICE-SAME: "-fcuda-is-device"
// DEVICE-SM52-SAME: "-target-cpu" "sm_52"
// THINLTOWPD-NOT: "-fwhole-program-vtables"
// DEVICE-SAME: "-o" "[[PTXFILE:[^"]*]]"
// DEVICE-NOSAVE-SAME: "-x" "cuda"
// DEVICE-SAVE-SAME: "-x" "ir"

// Match the call to ptxas (which assembles PTX to SASS).
// DEVICE:ptxas
// DEVICE-SM52-DAG: "--gpu-name" "sm_52"
// DEVICE-DAG: "--output-file" "[[CUBINFILE:[^"]*]]"
// DEVICE-DAG: "[[PTXFILE]]"

// Match another device-side compilation.
// DEVICE2: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// DEVICE2-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// DEVICE2-SAME: "-fcuda-is-device"
// DEVICE2-SM60-SAME: "-target-cpu" "sm_60"
// DEVICE2-SAME: "-o" "[[PTXFILE2:[^"]*]]"
// DEVICE2-SAME: "-x" "cuda"

// Match another call to ptxas.
// DEVICE2: ptxas
// DEVICE2-SM60-DAG: "--gpu-name" "sm_60"
// DEVICE2-DAG: "--output-file" "[[CUBINFILE2:[^"]*]]"
// DEVICE2-DAG: "[[PTXFILE2]]"

// Match no device-side compilation.
// NODEVICE-NOT: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// NODEVICE-NOT: "-fcuda-is-device"

// INCLUDES-DEVICE:fatbinary
// INCLUDES-DEVICE-DAG: "--create" "[[FATBINARY:[^"]*]]"
// INCLUDES-DEVICE-DAG: "--image=profile=sm_{{[0-9]+}},file=[[CUBINFILE]]"
// INCLUDES-DEVICE-DAG: "--image=profile=compute_{{[0-9]+}},file=[[PTXFILE]]"
// INCLUDES-DEVICE2-DAG: "--image=profile=sm_{{[0-9]+}},file=[[CUBINFILE2]]"
// INCLUDES-DEVICE2-DAG: "--image=profile=compute_{{[0-9]+}},file=[[PTXFILE2]]"

// Match host-side preprocessor job with -save-temps.
// HOST-SAVE: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-SAVE-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// HOST-SAVE-NOT: "-fcuda-is-device"
// HOST-SAVE-SAME: "-x" "cuda"

// Match host-side compilation.
// HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-SAME: "-aux-triple" "nvptx64-nvidia-cuda"
// THINLTOWPD-SAME: "-flto=thin"
// HOST-NOT: "-fcuda-is-device"
// There is only one GPU binary after combining it with fatbinary!
// INCLUDES-DEVICE2-NOT: "-fcuda-include-gpubinary"
// INCLUDES-DEVICE-SAME: "-fcuda-include-gpubinary" "[[FATBINARY]]"
// There is only one GPU binary after combining it with fatbinary.
// INCLUDES-DEVICE2-NOT: "-fcuda-include-gpubinary"
// THINLTOWPD-SAME: "-fwhole-program-vtables"
// HOST-SAME: "-o" "[[HOSTOUTPUT:[^"]*]]"
// HOST-NOSAVE-SAME: "-x" "cuda"
// HOST-SAVE-SAME: "-x" "cuda-cpp-output"

// Match external assembler that uses compilation output.
// HOST-AS: "-o" "{{.*}}.o" "[[HOSTOUTPUT]]"

// Match no GPU code inclusion.
// NOINCLUDES-DEVICE-NOT: "-fcuda-include-gpubinary"

// Match no host compilation.
// NOHOST-NOT: "-cc1" "-triple"
// NOHOST-NOT: "-x" "cuda"

// Match linker.
// LINK: "{{.*}}{{ld|link}}{{(.exe)?}}"
// LINK-SAME: "[[HOSTOUTPUT]]"

// Match no linker.
// NOLINK-NOT: "{{.*}}{{ld|link}}{{(.exe)?}}"

// FATBIN-COMMON:fatbinary
// FATBIN-COMMON: "--create" "[[FATBINARY:[^"]*]]"
// FATBIN-COMMON: "--image=profile=sm_52,file=
// PTX-SM52: "--image=profile=compute_52,file=
// NOPTX-SM52-NOT: "--image=profile=compute_52,file=
// FATBIN-COMMON: "--image=profile=sm_60,file=
// PTX-SM60: "--image=profile=compute_60,file=
// NOPTX-SM60-NOT: "--image=profile=compute_60,file=
