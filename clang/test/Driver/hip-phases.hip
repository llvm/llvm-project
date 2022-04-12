// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

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
// RDC-DAG: [[P12:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// RDC-DAG: [[P13:[0-9]+]]: assembler, {[[P12]]}, object, (host-[[T]])

// BIN-DAG: [[P3:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH:gfx803]])
// BIN-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH]])
// NRD-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH]])
// NRD-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH]])
// RDC-DAG: [[P7:[0-9]+]]: backend, {[[P5]]}, ir, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P8:[0-9]+]]: linker, {[[P7]]}, image, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P8]]}, image
// NRD-DAG: [[P10:[0-9]+]]: linker, {[[P9]]}, hip-fatbin, (device-[[T]])
// RDC-DAG: [[P10:[0-9]+]]: linker, {[[P9]]}, object, (device-[[T]])

// NRD-DAG: [[P11:[0-9]+]]: offload, "host-[[T]] (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-[[T]] (amdgcn-amd-amdhsa)" {[[P10]]}, ir
// RDC-DAG: [[P11:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa)" {[[P10]]}, object
// NRD-DAG: [[P12:[0-9]+]]: backend, {[[P11]]}, assembler, (host-[[T]])
// NRD-DAG: [[P13:[0-9]+]]: assembler, {[[P12]]}, object, (host-[[T]])
// NRD-DAG: [[P14:[0-9]+]]: linker, {[[P13]]}, image, (host-[[T]])
// RDC-DAG: [[P14:[0-9]+]]: linker, {[[P13]], [[P11]]}, image, (host-[[T]])

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
// Test two gpu architectures with complete compilation with -fno-gpu-rdc.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=NRD2,NCL2 %s

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s -c 2>&1 \
// RUN: | FileCheck -check-prefixes=NRD2 %s

// NRD2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// NRD2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// NRD2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])

// NRD2-DAG: [[P3:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH1:gfx803]])
// NRD2-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// NRD2-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH1]])
// NRD2-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH1]])
// NRD2-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH1]])
// NRD2-DAG: [[P8:[0-9]+]]: linker, {[[P7]]}, image, (device-[[T]], [[ARCH1]])
// NRD2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH1]])" {[[P8]]}, image

// NRD2-DAG: [[P10:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// NRD2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// NRD2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (device-[[T]], [[ARCH2]])
// NRD2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (device-[[T]], [[ARCH2]])
// NRD2-DAG: [[P14:[0-9]+]]: assembler, {[[P13]]}, object, (device-[[T]], [[ARCH2]])
// NRD2-DAG: [[P15:[0-9]+]]: linker, {[[P14]]}, image, (device-[[T]], [[ARCH2]])
// NRD2-DAG: [[P16:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P15]]}, image
// NRD2-DAG: [[P17:[0-9]+]]: linker, {[[P9]], [[P16]]}, hip-fatbin, (device-[[T]])
// NRD2-DAG: [[P18:[0-9]+]]: offload, "host-[[T]] (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-[[T]] (amdgcn-amd-amdhsa)" {[[P17]]}, ir
// NRD2-DAG: [[P19:[0-9]+]]: backend, {[[P18]]}, assembler, (host-[[T]])
// NRD2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-[[T]])
// NCL2-DAG: [[P21:[0-9]+]]: linker, {[[P20]]}, image, (host-[[T]])

//
// Test two gpu architectures with complete compilation with -fgpu-rdc.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s -fgpu-rdc 2>&1 \
// RUN: | FileCheck -check-prefixes=RDC2,RCL2 %s

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s -fgpu-rdc -c 2>&1 \
// RUN: | FileCheck -check-prefixes=RDC2,RC2 %s

// RCL2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// RCL2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// RCL2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// RCL2-DAG: [[P19:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// RCL2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-[[T]])

// RDC2-DAG: [[P3:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH1:gfx803]])
// RDC2-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// RDC2-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH1]])
// RDC2-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, ir, (device-[[T]], [[ARCH1]])
// RCL2-DAG: [[P8:[0-9]+]]: linker, {[[P6]]}, image, (device-[[T]], [[ARCH1]])
// RCL2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH1]])" {[[P8]]}, image
// RC2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH1]])" {[[P6]]}, ir

// RDC2-DAG: [[P10:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// RDC2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// RDC2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (device-[[T]], [[ARCH2]])
// RDC2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, ir, (device-[[T]], [[ARCH2]])
// RCL2-DAG: [[P15:[0-9]+]]: linker, {[[P13]]}, image, (device-[[T]], [[ARCH2]])
// RCL2-DAG: [[P16:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P15]]}, image
// RC2-DAG: [[P16:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P13]]}, ir

// RC2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// RC2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// RC2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// RC2-DAG: [[P19:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// RC2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-[[T]])

// RCL2-DAG: [[P17:[0-9]+]]: linker, {[[P9]], [[P16]]}, object, (device-[[T]])
// RCL2-DAG: [[P22:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa)" {[[P17]]}, object
// RCL2-DAG: [[P23:[0-9]+]]: linker, {[[P20]], [[P22]]}, image, (host-[[T]])
// RC2-DAG: [[P23:[0-9]+]]: clang-offload-bundler, {[[P9]], [[P16]], [[P20]]}, object, (host-[[T]])

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
// DBIN-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P5:[0-9]+]]: linker, {[[P4]]}, image, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P6:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P5]]}, image
// DBIN-DAG: [[P7:[0-9]+]]: linker, {[[P6]]}, hip-fatbin, (device-hip, )
// DBIN-DAG: [[P8:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:)" {[[P7]]}, hip-fatbin
// DBIN-NOT: host
//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only -S --no-gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM %s
// DASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P3]]}, assembler
// DASM-NOT: clang-offload-bundler
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
// DBIN2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P5:[0-9]+]]: linker, {[[P4]]}, image, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P6:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P5]]}, image
// DBIN2-DAG: [[P7:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// DBIN2-DAG: [[P8:[0-9]+]]: preprocessor, {[[P7]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P9:[0-9]+]]: compiler, {[[P8]]}, ir, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P10:[0-9]+]]: backend, {[[P9]]}, assembler, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P11:[0-9]+]]: assembler, {[[P10]]}, object, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P12:[0-9]+]]: linker, {[[P11]]}, image, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P13:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P12]]}, image
// DBIN2-DAG: [[P14:[0-9]+]]: linker, {[[P6]], [[P13]]}, hip-fatbin, (device-hip, )
// DBIN2-DAG: [[P15:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:)" {[[P14]]}, hip-fatbin
// DBIN2-NOT: host
//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -S -o %t.s 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM2,DASM2-NOBUNDLE %s
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -S -o %t.s --no-gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM2,DASM2-NOBUNDLE %s
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM2,DASM2-NOBUNDLE %s
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -S --gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM2,DASM2-BUNDLE %s
// DASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// DASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P3]]}, assembler
// DASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// DASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P8]]}, assembler
// DASM2-BUNDLE: [[P10:[0-9]+]]: clang-offload-bundler, {[[P4]], [[P9]]}, assembler, (device-hip, )
// DASM2-NOBUNDLE-NOT: clang-offload-bundler, {[[P4]], [[P9]]}, assembler, (device-hip, )
// DASM2-NOT: host

//
// Test linking two objects with two gpu architectures.
//
// RUN: touch %T/obj1.o
// RUN: touch %T/obj2.o
//
// RUN: %clang -target x86_64-unknown-linux-gnu -ccc-print-phases --hip-link \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %T/obj1.o %T/obj2.o 2>&1 \
// RUN: | FileCheck -check-prefixes=L2,NL2 %s
//
// RUN: %clang -target x86_64-unknown-linux-gnu -ccc-print-phases --hip-link \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %T/obj1.o %T/obj2.o \
// RUN: -fgpu-rdc 2>&1 | FileCheck -check-prefixes=L2,RL2,RL2-EM %s
//
// RUN: %clang -target x86_64-unknown-linux-gnu -ccc-print-phases --hip-link \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %T/obj1.o %T/obj2.o \
// RUN: -fgpu-rdc --cuda-device-only 2>&1 | FileCheck -check-prefixes=L2,RL2,RL2-DEV %s

// RUN: %clang -target x86_64-unknown-linux-gnu -ccc-print-phases --hip-link \
// RUN: --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %T/obj1.o %T/obj2.o \
// RUN: -fgpu-rdc --cuda-device-only --no-gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=L2,RL2,RL2-NB %s

// L2-DAG: [[P0:[0-9]+]]: input, "{{.*}}obj1.o", object
// RL2-DAG: [[P1:[0-9]+]]: clang-offload-unbundler, {[[P0]]}, object
// L2-DAG: [[P2:[0-9]+]]: input, "{{.*}}obj2.o", object
// RL2-DAG: [[P3:[0-9]+]]: clang-offload-unbundler, {[[P2]]}, object

// RL2-DAG: [[P4:[0-9]+]]: linker, {[[P1]], [[P3]]}, image, (device-[[T:hip]], [[ARCH1:gfx803]])
// RL2-DAG: [[P5:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH1]])" {[[P4]]}, image
// RL2-DAG: [[P6:[0-9]+]]: linker, {[[P1]], [[P3]]}, image, (device-[[T]], [[ARCH2:gfx900]])
// RL2-DAG: [[P7:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P6]]}, image
// RL2-DEV-DAG: [[P8:[0-9]+]]: linker, {[[P5]], [[P7]]}, hip-fatbin, (device-[[T]])
// RL2-DEV-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa)" {[[P8]]}, hip-fatbin
// RL2-EM-DAG: [[P8:[0-9]+]]: linker, {[[P5]], [[P7]]}, object, (device-[[T]])
// RL2-EM-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa)" {[[P8]]}, object
// RL2-NB-NOT: linker
// RL2-NB-NOT: offload

// NL2-DAG: [[P4:[0-9]+]]: linker, {[[P0]], [[P2]]}, image
// RL2-EM-DAG: [[P4:[0-9]+]]: linker, {[[P1]], [[P3]], [[P9]]}, image, (host-[[T]])
// RL2-DEV-NOT: linker

// Test one gpu architectures up to the preprocessor expansion output phase in device-only
// compilation mode. no bundle.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 %s \
// RUN: --cuda-device-only -E 2>&1 \
// RUN: | FileCheck -check-prefixes=PPE,PPEN %s

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 %s \
// RUN: --cuda-device-only -E --no-gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=PPE,PPEN %s

// Test one gpu architectures up to the preprocessor expansion output phase in device-only
// compilation mode. bundle.

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 %s \
// RUN: --cuda-device-only -E --gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=PPE,PPEB %s

// Test two gpu architectures up to the preprocessor expansion output phase in device-only
// compilation mode. no bundle.

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -E 2>&1 \
// RUN: | FileCheck -check-prefixes=PPE2,PPE2N %s

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -E --no-gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=PPE2,PPE2N %s

// Test two gpu architectures up to the preprocessor expansion output phase in device-only
// compilation mode. bundle.

// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -E --gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=PPE2,PPE2B %s

// Test one gpu architectures up to the LLVM IR output phase in device-only
// compilation mode. no bundle.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 %s \
// RUN: --cuda-device-only -c -emit-llvm 2>&1 \
// RUN: | FileCheck -check-prefixes=LLVM %s

// Test two gpu architectures up to the LLVM IR output phase in device-only
// compilation mode. bundle.
//
// RUN: %clang -x hip -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -c -emit-llvm -o %t.bc --gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=LLVM2 %s

// Test two gpu architectures up to the LLVM IR output phase in device-only
// compilation mode with bundled preprocessor expansion as input. bundle.
//
// RUN: %clang -x hip-cpp-output -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 %s \
// RUN: --cuda-device-only -c -emit-llvm -o %t.bc --gpu-bundle-output 2>&1 \
// RUN: | FileCheck -check-prefixes=PPELLVM2 %s

// PPE-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// PPE-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// PPE-DAG: [[P2:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P1]]}, [[T]]-cpp-output
// PPEB-DAG: [[P3:[0-9]+]]: clang-offload-bundler, {[[P2]]}, [[T]]-cpp-output, (device-hip, )
// PPEN-NOT: clang-offload-bundler, {{.*}}, [[T]]-cpp-output, (device-hip, )
// PPE-NOT: host

// PPE2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// PPE2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// PPE2-DAG: [[P2:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P1]]}, [[T]]-cpp-output
// PPE2-DAG: [[P5:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// PPE2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// PPE2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P6]]}, [[T]]-cpp-output
// PPE2B-DAG: [[P10:[0-9]+]]: clang-offload-bundler, {[[P2]], [[P9]]}, [[T]]-cpp-output, (device-hip, )
// PPE2N-NOT: clang-offload-bundler, {{.*}}, [[T]]-cpp-output, (device-hip, )
// PPE2-NOT: host

// LLVM-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// LLVM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// LLVM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// LLVM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, ir, (device-[[T]], [[ARCH]])
// LLVM-NOT: clang-offload-bundler
// LLVM-NOT: host

// LLVM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// LLVM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// LLVM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// LLVM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, ir, (device-[[T]], [[ARCH]])
// LLVM2-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P3]]}, ir
// LLVM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH2:gfx900]])
// LLVM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// LLVM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// LLVM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, ir, (device-[[T]], [[ARCH2]])
// LLVM2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P8]]}, ir
// LLVM2-DAG: [[P10:[0-9]+]]: clang-offload-bundler, {[[P4]], [[P9]]}, ir, (device-hip, )
// LLVM2-NOT: host

// PPELLVM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]]-cpp-output
// PPELLVM2-DAG: [[P1:[0-9]+]]: clang-offload-unbundler, {[[P0]]}, hip-cpp-output
// PPELLVM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH:gfx803]])
// PPELLVM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, ir, (device-[[T]], [[ARCH]])
// PPELLVM2-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH]])" {[[P3]]}, ir
// PPELLVM2-DAG: [[P7:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH2:gfx900]])
// PPELLVM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, ir, (device-[[T]], [[ARCH2]])
// PPELLVM2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] (amdgcn-amd-amdhsa:[[ARCH2]])" {[[P8]]}, ir
// PPELLVM2-DAG: [[P10:[0-9]+]]: clang-offload-bundler, {[[P4]], [[P9]]}, ir, (device-hip, )
// PPELLVM2-NOT: host

// Test mixed HIP and C++ compilation. HIP program should have HIP offload kind.
// C++ program should have no offload kind.

// Test compile empty.hip and empty.cpp.
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -c %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED %s
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -c %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED-NEG %s

// Test compile and link empty.hip and empty.cpp.
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED %s
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED-NEG %s

// Test compile and link empty.hip and empty.cpp with --hip-link -fgpu-rdc.
// RUN: %clang -target x86_64-unknown-linux-gnu --hip-link -fgpu-rdc \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED %s
// RUN: %clang -target x86_64-unknown-linux-gnu --hip-link -fgpu-rdc \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED-NEG %s

// Test compile and link -x hip empty.hip and -x c++ empty.cpp.
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -x hip %S/Inputs/empty.hip -x c++ %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED %s
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -x hip %S/Inputs/empty.hip -x c++ %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED-NEG %s

// Test compile and link -x hip empty.hip and empty.cpp.
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -x hip %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -x hip %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED2-NEG %s

// Test compile and link empty.hip and -x hip empty.cpp.
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: %S/Inputs/empty.hip -x hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu \
// RUN: -ccc-print-phases --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN: -x hip %S/Inputs/empty.hip %S/Inputs/empty.cpp 2>&1 | FileCheck -check-prefixes=MIXED2-NEG %s

// MIXED-DAG: input, "{{.*}}empty.hip", hip, (host-hip)
// MIXED-DAG: input, "{{.*}}empty.hip", hip, (device-hip, gfx803)
// MIXED-DAG: input, "{{.*}}empty.hip", hip, (device-hip, gfx900)
// MIXED-DAG: input, "{{.*}}empty.cpp", c++
// MIXED-NEG-NOT: input, "{{.*}}empty.cpp", c++, (host-hip)
// MIXED-NEG-NOT: input, "{{.*}}empty.cpp", c++, (device-hip

// MIXED2-DAG: input, "{{.*}}empty.hip", hip, (host-hip)
// MIXED2-DAG: input, "{{.*}}empty.hip", hip, (device-hip, gfx803)
// MIXED2-DAG: input, "{{.*}}empty.hip", hip, (device-hip, gfx900)
// MIXED2-DAG: input, "{{.*}}empty.cpp", hip, (host-hip)
// MIXED2-DAG: input, "{{.*}}empty.cpp", hip, (device-hip, gfx803)
// MIXED2-DAG: input, "{{.*}}empty.cpp", hip, (device-hip, gfx900)
// MIXED2-NEG-NOT: input, "{{.*}}empty.cpp", c++
