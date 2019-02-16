// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

// REQUIRES: clang-driver
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target
//
// Test single gpu architecture with complete compilation.
//
// Test CUDA NVPTX phases.
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,BIN_NV %s
//
// Test HIP AMDGPU -fgpu-rdc phases.
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 -fgpu-rdc %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,BIN_AMD,BIN_AMD_RDC %s
//
// Test HIP AMDGPU -fno-gpu-rdc phases (default).
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,BIN_AMD,BIN_AMD_NRDC %s
//
// BIN_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// BIN_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (host-[[T]])
// BIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// BIN_NV-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH:sm_30]])
// BIN_AMD-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH:gfx803]])
// BIN-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH]])
// BIN_NV-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH]])
// BIN_NV-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH]])
// BIN_NV-DAG: [[P8:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P7]]}, object
// BIN_NV-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH]])" {[[P6]]}, assembler
// BIN_NV-DAG: [[P10:[0-9]+]]: linker, {[[P8]], [[P9]]}, cuda-fatbin, (device-[[T]])
// BIN_NV-DAG: [[P11:[0-9]+]]: offload, "host-[[T]] (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-[[T]] ([[TRIPLE]])" {[[P10]]}, ir
// BIN_NV-DAG: [[P12:[0-9]+]]: backend, {[[P11]]}, assembler, (host-[[T]])
// BIN_AMD_RDC-DAG: [[P12:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// BIN_AMD_NRDC-DAG: [[P6:[0-9]+]]: linker, {[[P5]]}, image, (device-hip, [[ARCH]])
// BIN_AMD_NRDC-DAG: [[P7:[0-9]+]]: offload, "device-hip (amdgcn-amd-amdhsa:[[ARCH]])" {[[P6]]}, image
// BIN_AMD_NRDC-DAG: [[P8:[0-9]+]]: linker, {[[P7]]}, hip-fatbin, (device-hip)
// BIN_AMD_NRDC-DAG: [[P11:[0-9]+]]: offload, "host-hip (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-hip (amdgcn-amd-amdhsa)" {[[P8]]}, ir
// BIN_AMD_NRDC-DAG: [[P12:[0-9]+]]: backend, {[[P11]]}, assembler, (host-[[T]])
// BIN-DAG: [[P13:[0-9]+]]: assembler, {[[P12]]}, object, (host-[[T]])
// BIN-DAG: [[P14:[0-9]+]]: linker, {[[P13]]}, image, (host-[[T]])
// BIN_AMD_RDC-DAG: [[P15:[0-9]+]]: linker, {[[P5]]}, image, (device-[[T]], [[ARCH]])
// BIN_AMD_RDC-DAG: [[P16:[0-9]+]]: offload, "host-[[T]] (powerpc64le-ibm-linux-gnu)" {[[P14]]},
// BIN_AMD_RDC-DAG-SAME:  "device-[[T]] ([[TRIPLE:amdgcn-amd-amdhsa]]:[[ARCH]])" {[[P15]]}, object

//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM,ASM_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 -fgpu-rdc %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM,ASM_AMD %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 -fcuda-rdc %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM,ASM_AMD %s
// ASM_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// ASM_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// ASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// ASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// ASM_NV-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// ASM_NV-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda|amdgcn-amd-amdhsa]]:[[ARCH]])" {[[P3]]}, assembler
// ASM-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (host-[[T]])
// ASM-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (host-[[T]])
// ASM-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (host-[[T]])
// ASM-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (host-[[T]])

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2,BIN2_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 -fgpu-rdc %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2,BIN2_AMD %s
// BIN2_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// BIN2_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (host-[[T]])
// BIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// BIN2-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH1:sm_30|gfx803]])
// BIN2-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH1]])
// BIN2_NV-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH1]])
// BIN2_NV-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH1]])
// BIN2_NV-DAG: [[P8:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH1]])" {[[P7]]}, object
// BIN2_NV-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH1]])" {[[P6]]}, assembler
// BIN2-DAG: [[P10:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35|gfx900]])
// BIN2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (device-[[T]], [[ARCH2]])
// BIN2_NV-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (device-[[T]], [[ARCH2]])
// BIN2_NV-DAG: [[P14:[0-9]+]]: assembler, {[[P13]]}, object, (device-[[T]], [[ARCH2]])
// BIN2_NV-DAG: [[P15:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P14]]}, object
// BIN2_NV-DAG: [[P16:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P13]]}, assembler
// BIN2_NV-DAG: [[P17:[0-9]+]]: linker, {[[P8]], [[P9]], [[P15]], [[P16]]}, cuda-fatbin, (device-[[T]])
// BIN2_NV-DAG: [[P18:[0-9]+]]: offload, "host-[[T]] (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-[[T]] ([[TRIPLE]])" {[[P17]]}, ir
// BIN2_NV-DAG: [[P19:[0-9]+]]: backend, {[[P18]]}, assembler, (host-[[T]])
// BIN2_AMD-DAG: [[P19:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// BIN2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-[[T]])
// BIN2-DAG: [[P21:[0-9]+]]: linker, {[[P20]]}, image, (host-[[T]])
// BIN2_AMD-DAG: [[P22:[0-9]+]]: linker, {[[P5]]}, image, (device-[[T]], [[ARCH1]])
// BIN2_AMD-DAG: [[P23:[0-9]+]]: linker, {[[P12]]}, image, (device-[[T]], [[ARCH2]])
// BIN2_AMD-DAG: [[P24:[0-9]+]]: offload, "host-[[T]] (powerpc64le-ibm-linux-gnu)" {[[P21]]},
// BIN2_AMD-DAG-SAME:  "device-[[T]] ([[TRIPLE:amdgcn-amd-amdhsa]]:[[ARCH1]])" {[[P22]]},
// BIN2_AMD-DAG-SAME:  "device-[[T]] ([[TRIPLE:amdgcn-amd-amdhsa]]:[[ARCH2]])" {[[P23]]}, object

//
// Test two gpu architecturess up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM2,ASM2_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 -fgpu-rdc %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM2,ASM2_AMD %s
// ASM2_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH1:sm_30]])
// ASM2_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (device-[[T]], [[ARCH1:gfx803]])
// ASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// ASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH1]])
// ASM2_NV-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH1]])
// ASM2_NV-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda|amdgcn-amd-amdhsa]]:[[ARCH1]])" {[[P3]]}, assembler
// ASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35|gfx900]])
// ASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// ASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// ASM2_NV-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-[[T]], [[ARCH2]])
// ASM2_NV-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P8]]}, assembler
// ASM2-DAG: [[P10:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (host-[[T]])
// ASM2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (host-[[T]])
// ASM2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (host-[[T]])
// ASM2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (host-[[T]])

//
// Test single gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN,HBIN_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN,HBIN_AMD %s
// HBIN_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// HBIN_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (host-[[T]])
// HBIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HBIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HBIN-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HBIN-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (host-[[T]])
// HBIN-DAG: [[P5:[0-9]+]]: linker, {[[P4]]}, image, (host-[[T]])
// HBIN-NOT: device
//
// Test single gpu architecture up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=HASM,HASM_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=HASM,HASM_AMD %s
// HASM_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// HASM_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (host-[[T]])
// HASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HASM-NOT: device

//
// Test two gpu architectures with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN2,HBIN2_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN2,HBIN2_AMD %s
// HBIN2_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// HBIN2_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (host-[[T]])
// HBIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HBIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HBIN2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HBIN2-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (host-[[T]])
// HBIN2-DAG: [[P5:[0-9]+]]: linker, {[[P4]]}, image, (host-[[T]])
// HBIN2-NOT: device

//
// Test two gpu architectures up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only -S \
// RUN: 2>&1 | FileCheck -check-prefixes=HASM2,HASM2_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --cuda-host-only -S \
// RUN: 2>&1 | FileCheck -check-prefixes=HASM2,HASM2_AMD %s
// HASM2_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// HASM2_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (host-[[T]])
// HASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HASM2-NOT: device

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefixes=DBIN,DBIN_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefixes=DBIN,DBIN_AMD %s
// DBIN_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DBIN_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DBIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DBIN_NV-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DBIN_NV-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// DBIN_NV-DAG: [[P5:[0-9]+]]: offload, "device-[[T]] (nvptx64-nvidia-cuda:[[ARCH]])" {[[P4]]}, object
// DBIN-NOT: host
//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM,DASM_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM,DASM_AMD %s
// DASM_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DASM_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM_NV-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DASM_NV-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda|amdgcn-amd-amdhsa]]:[[ARCH]])" {[[P3]]}, assembler
// DASM-NOT: host

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefixes=DBIN2,DBIN2_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --cuda-device-only \
// RUN: 2>&1 | FileCheck -check-prefixes=DBIN2,DBIN2_AMD %s
// DBIN2_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DBIN2_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DBIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DBIN2_NV-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DBIN2_NV-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// DBIN2_NV-DAG: [[P5:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P4]]}, object
// DBIN2-DAG: [[P6:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35|gfx900]])
// DBIN2-DAG: [[P7:[0-9]+]]: preprocessor, {[[P6]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P8:[0-9]+]]: compiler, {[[P7]]}, ir, (device-[[T]], [[ARCH2]])
// DBIN2_NV-DAG: [[P9:[0-9]+]]: backend, {[[P8]]}, assembler, (device-[[T]], [[ARCH2]])
// DBIN2_NV-DAG: [[P10:[0-9]+]]: assembler, {[[P9]]}, object, (device-[[T]], [[ARCH2]])
// DBIN2_NV-DAG: [[P11:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P10]]}, object
// DBIN2-NOT: host
//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only -S \
// RUN: 2>&1 | FileCheck -check-prefixes=DASM2,DASM2_NV %s
// RUN: %clang -x hip -target powerpc64le-ibm-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM2,DASM2_AMD %s
// DASM2_NV-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DASM2_AMD-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM2_NV-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DASM2_NV-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda|amdgcn-amd-amdhsa]]:[[ARCH]])" {[[P3]]}, assembler
// DASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35|gfx900]])
// DASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// DASM2_NV-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-[[T]], [[ARCH2]])
// DASM2_NV-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P8]]}, assembler
// DASM2-NOT: host
