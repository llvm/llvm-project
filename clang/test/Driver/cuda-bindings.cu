// Tests the bindings generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.
// It parallels cuda-phases.cu test, but verifies whether output file is temporary or not.

// It's hard to check whether file name is temporary in a portable
// way. Instead we check whether we've generated a permanent name on
// device side, which appends '-device-cuda-<triple>' suffix.

// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

//
// Test single gpu architecture with complete compilation.
// No intermediary device files should have "-device-cuda..." in the name.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefix=BIN %s
// BIN: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "nvptx64-nvidia-cuda" - "NVPTX::Linker",{{.*}} output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}}  output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "a.out"

//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings --cuda-gpu-arch=sm_30 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM %s
// ASM-DAG: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"
// ASM-DAG: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}} output: "cuda-bindings.s"

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2,AOUT %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:       --offload-arch=sm_30,sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2,AOUT %s
// .. same, but with explicitly specified output.
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:       --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -o %t/out 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2,TOUT %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --offload-arch=sm_30,sm_35 %s -o %t/out 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2,TOUT %s
// BIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Linker",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}}  output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// AOUT: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "a.out"
// TOUT: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "{{.*}}/out"

// .. same, but with -fsyntax-only
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:       --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix=SYN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:        --offload-arch=sm_30,sm_35 %s -o %t/out 2>&1 \
// RUN: | FileCheck -check-prefix=SYN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:       --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix=SYN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:        --offload-arch=sm_30,sm_35 %s -o %t/out 2>&1 \
// RUN: | FileCheck -check-prefix=SYN %s
// SYN-NOT: inputs:
// SYN: # "powerpc64le-ibm-linux-gnu" - "clang", inputs: [{{.*}}], output: (nothing)
// SYN-NEXT: # "nvptx64-nvidia-cuda" - "clang", inputs: [{{.*}}], output: (nothing)
// SYN-NEXT: # "nvptx64-nvidia-cuda" - "clang", inputs: [{{.*}}], output: (nothing)
// SYN-NOT: inputs

// .. and with --offload-new-driver
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:       --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 --offload-new-driver %s 2>&1 \
// RUN: | FileCheck -check-prefix=NDSYN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:        --offload-arch=sm_30,sm_35 %s --offload-new-driver -o %t/out 2>&1 \
// RUN: | FileCheck -check-prefix=NDSYN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:       --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --offload-new-driver 2>&1 \
// RUN: | FileCheck -check-prefix=NDSYN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings -fsyntax-only \
// RUN:        --offload-arch=sm_30,sm_35 %s --offload-new-driver -o %t/out 2>&1 \
// RUN: | FileCheck -check-prefix=NDSYN %s
// NDSYN-NOT: inputs:
// NDSYN: # "nvptx64-nvidia-cuda" - "clang", inputs: [{{.*}}], output: (nothing)
// NDSYN-NEXT: # "nvptx64-nvidia-cuda" - "clang", inputs: [{{.*}}], output: (nothing)
// NDSYN-NEXT: # "powerpc64le-ibm-linux-gnu" - "clang", inputs: [{{.*}}], output: (nothing)
// NDSYN-NOT: inputs:


//
// Test two gpu architectures up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM2 %s
// ASM2-DAG: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"
// ASM2-DAG: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_35.s"
// ASM2-DAG: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}} output: "cuda-bindings.s"

//
// Test one or more gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN %s
// HBIN: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}}  output:
// HBIN-NOT: cuda-bindings-device-cuda-nvptx64
// HBIN: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "a.out"

//
// Test one or more gpu architecture up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM %s
// HASM: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}} output: "cuda-bindings.s"

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN %s
// DBIN: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// DBIN-NOT: cuda-bindings-device-cuda-nvptx64
// DBIN: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.o"

//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM %s
// DASM: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN2 %s
// DBIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// DBIN2-NOT: cuda-bindings-device-cuda-nvptx64
// DBIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.o"
// DBIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// DBIN2-NOT: cuda-bindings-device-cuda-nvptx64
// DBIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_35.o"

//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM2 %s
// DASM2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"
// DASM2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_35.s"

//
// Ensure we output the user's specified name in device-only mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -### \
// RUN:        --cuda-gpu-arch=sm_52 --cuda-device-only -c -o foo.o %s 2>&1 \
// RUN: | FileCheck -check-prefix=D_ONLY %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -### --offload-new-driver \
// RUN:        --cuda-gpu-arch=sm_52 --cuda-device-only -c -o foo.o %s 2>&1 \
// RUN: | FileCheck -check-prefix=D_ONLY %s
// D_ONLY: "foo.o"

//
// Check to make sure we can generate multiple outputs for device-only
// compilation and fail with '-o'.
//
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu --offload-new-driver -ccc-print-bindings \
// RUN:        --offload-arch=sm_70 --offload-arch=sm_52 --offload-device-only -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY %s
//      MULTI-D-ONLY: # "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX_70:.+]]"
// MULTI-D-ONLY-NEXT: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_70]]"], output: "[[CUBIN_70:.+]]"
// MULTI-D-ONLY-NEXT: # "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]"], output: "[[PTX_52:.+]]"
// MULTI-D-ONLY-NEXT: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_52]]"], output: "[[CUBIN_52:.+]]"
//
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu --offload-new-driver -ccc-print-bindings \
// RUN:        --offload-arch=sm_70 --offload-arch=sm_52 --offload-device-only -c -o %t %s 2>&1 \
// RUN: | FileCheck -check-prefix=MULTI-D-ONLY-O %s
// MULTI-D-ONLY-O: error: cannot specify -o when generating multiple output files

//
// Check to ensure that we can use '-fsyntax-only' for CUDA output with the new
// driver.
// 
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu --offload-new-driver \
// RUN:        -fsyntax-only --offload-arch=sm_70 --offload-arch=sm_52 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=SYNTAX-ONLY %s
// SYNTAX-ONLY: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-fsyntax-only"
// SYNTAX-ONLY: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-fsyntax-only"
// SYNTAX-ONLY: "-cc1" "-triple" "powerpc64le-ibm-linux-gnu"{{.*}}"-fsyntax-only"

//
// Check to ensure that we can use '-save-temps' when operating in RDC-mode.
//
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu -save-temps --offload-new-driver \
// RUN:        -fgpu-rdc --offload-arch=sm_70 --offload-arch=sm_52 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=SAVE-TEMPS %s
// SAVE-TEMPS: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_52"
// SAVE-TEMPS: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}}"-target-cpu" "sm_70"
// SAVE-TEMPS: "-cc1" "-triple" "powerpc64le-ibm-linux-gnu"

//
// Check to ensure that we cannot use '-foffload' when not operating in RDC-mode.
//
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu -fno-gpu-rdc --offload-new-driver \
// RUN:        -foffload-lto --offload-arch=sm_70 --offload-arch=sm_52 -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=LTO-NO-RDC %s
// LTO-NO-RDC: error: unsupported option '-foffload-lto' for language mode '-fno-gpu-rdc'
