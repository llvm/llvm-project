// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
//
// Test single gpu architecture with complete compilation.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,NRD %s
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 -fgpu-rdc %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,RDC %s
//
// BIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// BIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])

// BIN-DAG: [[P3:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH:gfx803]])
// BIN-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH]])
// NRD-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH]])
// NRD-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH]])
// NRD-DAG: [[P8:[0-9]+]]: linker, {[[P7]]}, image, (device-hip, [[ARCH]])
// NRD-DAG: [[P9:[0-9]+]]: offload, "device-hip (amdgcn-amd-amdhsa:[[ARCH]])" {[[P8]]}, image
// NRD-DAG: [[P10:[0-9]+]]: linker, {[[P9]]}, hip-fatbin, (device-hip)
// RDC-DAG: [[P10:[0-9]+]]: linker, {[[P5]]}, image, (device-hip, [[ARCH]])

// NRD-DAG: [[P12:[0-9]+]]: offload, "host-hip (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-hip (amdgcn-amd-amdhsa)" {[[P10]]}, ir

// NRD-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (host-[[T]])
// RDC-DAG: [[P13:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// BIN-DAG: [[P14:[0-9]+]]: assembler, {[[P13]]}, object, (host-[[T]])
// BIN-DAG: [[P15:[0-9]+]]: linker, {[[P14]]}, image, (host-[[T]])
// RDC-DAG: [[P16:[0-9]+]]: offload, "host-hip (x86_64-unknown-linux-gnu)" {[[P15]]}, "device-hip (amdgcn-amd-amdhsa:gfx803)" {[[P10]]}, image
//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM %s
// ASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// ASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// ASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])

// ASM-DAG: [[P5:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (host-[[T]])
// ASM-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (host-[[T]])
// ASM-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (host-[[T]])
// ASM-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (host-[[T]])

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2 %s

// BIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// BIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])

// BIN2-DAG: [[P3:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH1:gfx803]])
// BIN2-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P8:[0-9]+]]: linker, {[[P7]]}, image, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P9:[0-9]+]]: offload, "device-hip (amdgcn-amd-amdhsa:[[ARCH1]])" {[[P8]]}, image

// BIN2-DAG: [[P10:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// BIN2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P14:[0-9]+]]: assembler, {[[P13]]}, object, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P15:[0-9]+]]: linker, {[[P14]]}, image, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P16:[0-9]+]]: offload, "device-hip (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P15]]}, image

// BIN2-DAG: [[P17:[0-9]+]]: linker, {[[P9]], [[P16]]}, hip-fatbin, (device-hip)

// BIN2-DAG: [[P18:[0-9]+]]: offload, "host-hip (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-hip (amdgcn-amd-amdhsa)" {[[P17]]}, ir
// BIN2-DAG: [[P19:[0-9]+]]: backend, {[[P18]]}, assembler, (host-[[T]])
// BIN2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-[[T]])
// BIN2-DAG: [[P21:[0-9]+]]: linker, {[[P20]]}, image, (host-[[T]])

//
// Test two gpu architecturess up to the assemble phase.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM2 %s
// ASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH1:gfx803]])
// ASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// ASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH1]])
// ASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:sm_35|gfx900]])
// ASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// ASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// ASM2-DAG: [[P10:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (host-[[T]])
// ASM2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (host-[[T]])
// ASM2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (host-[[T]])
// ASM2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (host-[[T]])

//
// Test single gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN %s
// HBIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
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
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=HASM %s
// HASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// HASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HASM-NOT: device

//
// Test two gpu architectures with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN2 %s
// HBIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
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
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --cuda-host-only -S \
// RUN: 2>&1 | FileCheck -check-prefixes=HASM2 %s
// HASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// HASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HASM2-NOT: device

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefixes=DBIN %s
// DBIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DBIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DBIN-NOT: host
//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM %s
// DASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM-NOT: host

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s --cuda-device-only \
// RUN: 2>&1 | FileCheck -check-prefixes=DBIN2 %s
// DBIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DBIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P6:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// DBIN2-DAG: [[P7:[0-9]+]]: preprocessor, {[[P6]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P8:[0-9]+]]: compiler, {[[P7]]}, ir, (device-[[T]], [[ARCH2]])
// DBIN2-NOT: host
//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM2 %s
// DASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// DASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// DASM2-NOT: host
