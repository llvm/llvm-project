// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

// REQUIRES: clang-driver
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

//
// Test single gpu architecture with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefix=BIN %s
// BIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// BIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (host-cuda)
// BIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-cuda)
// BIN-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// BIN-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, cuda-cpp-output, (device-cuda, sm_30)
// BIN-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-cuda, sm_30)
// BIN-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-cuda, sm_30)
// BIN-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-cuda, sm_30)
// BIN-DAG: [[P8:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P7]]}, object
// BIN-DAG: [[P9:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P6]]}, assembler
// BIN-DAG: [[P10:[0-9]+]]: linker, {[[P8]], [[P9]]}, cuda-fatbin, (device-cuda)
// BIN-DAG: [[P11:[0-9]+]]: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-cuda (nvptx64-nvidia-cuda)" {[[P10]]}, ir
// BIN-DAG: [[P12:[0-9]+]]: backend, {[[P11]]}, assembler, (host-cuda)
// BIN-DAG: [[P13:[0-9]+]]: assembler, {[[P12]]}, object, (host-cuda)
// BIN-DAG: [[P14:[0-9]+]]: linker, {[[P13]]}, image, (host-cuda)

//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM %s
// ASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// ASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (device-cuda, sm_30)
// ASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-cuda, sm_30)
// ASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-cuda, sm_30)
// ASM-DAG: [[P4:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P3]]}, assembler
// ASM-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// ASM-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, cuda-cpp-output, (host-cuda)
// ASM-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (host-cuda)
// ASM-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (host-cuda)

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix=BIN2 %s
// BIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// BIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (host-cuda)
// BIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-cuda)
// BIN2-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// BIN2-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, cuda-cpp-output, (device-cuda, sm_30)
// BIN2-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-cuda, sm_30)
// BIN2-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-cuda, sm_30)
// BIN2-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-cuda, sm_30)
// BIN2-DAG: [[P8:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P7]]}, object
// BIN2-DAG: [[P9:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P6]]}, assembler
// BIN2-DAG: [[P10:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// BIN2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, cuda-cpp-output, (device-cuda, sm_35)
// BIN2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (device-cuda, sm_35)
// BIN2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (device-cuda, sm_35)
// BIN2-DAG: [[P14:[0-9]+]]: assembler, {[[P13]]}, object, (device-cuda, sm_35)
// BIN2-DAG: [[P15:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {[[P14]]}, object
// BIN2-DAG: [[P16:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {[[P13]]}, assembler
// BIN2-DAG: [[P17:[0-9]+]]: linker, {[[P8]], [[P9]], [[P15]], [[P16]]}, cuda-fatbin, (device-cuda)
// BIN2-DAG: [[P18:[0-9]+]]: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-cuda (nvptx64-nvidia-cuda)" {[[P17]]}, ir
// BIN2-DAG: [[P19:[0-9]+]]: backend, {[[P18]]}, assembler, (host-cuda)
// BIN2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-cuda)
// BIN2-DAG: [[P21:[0-9]+]]: linker, {[[P20]]}, image, (host-cuda)

//
// Test two gpu architecturess up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM2 %s
// ASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// ASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (device-cuda, sm_30)
// ASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-cuda, sm_30)
// ASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-cuda, sm_30)
// ASM2-DAG: [[P4:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P3]]}, assembler
// ASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// ASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, cuda-cpp-output, (device-cuda, sm_35)
// ASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-cuda, sm_35)
// ASM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-cuda, sm_35)
// ASM2-DAG: [[P9:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {[[P8]]}, assembler
// ASM2-DAG: [[P10:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// ASM2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, cuda-cpp-output, (host-cuda)
// ASM2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (host-cuda)
// ASM2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (host-cuda)

//
// Test single gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN %s
// HBIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HBIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (host-cuda)
// HBIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-cuda)
// HBIN-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-cuda)
// HBIN-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (host-cuda)
// HBIN-DAG: [[P5:[0-9]+]]: linker, {[[P4]]}, image, (host-cuda)
//
// Test single gpu architecture up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM %s
// HASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (host-cuda)
// HASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-cuda)
// HASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-cuda)

//
// Test two gpu architectures with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN2 %s
// HBIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HBIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (host-cuda)
// HBIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-cuda)
// HBIN2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-cuda)
// HBIN2-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (host-cuda)
// HBIN2-DAG: [[P5:[0-9]+]]: linker, {[[P4]]}, image, (host-cuda)

//
// Test two gpu architectures up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM2 %s
// HASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (host-cuda)
// HASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-cuda)
// HASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-cuda)

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN %s
// DBIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DBIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (device-cuda, sm_30)
// DBIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-cuda, sm_30)
// DBIN-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-cuda, sm_30)
// DBIN-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-cuda, sm_30)
// DBIN-DAG: [[P5:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P4]]}, object

//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM %s
// DASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (device-cuda, sm_30)
// DASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-cuda, sm_30)
// DASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-cuda, sm_30)
// DASM-DAG: [[P4:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P3]]}, assembler

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN2 %s
// DBIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DBIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (device-cuda, sm_30)
// DBIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-cuda, sm_30)
// DBIN2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-cuda, sm_30)
// DBIN2-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-cuda, sm_30)
// DBIN2-DAG: [[P5:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P4]]}, object
// DBIN2-DAG: [[P6:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// DBIN2-DAG: [[P7:[0-9]+]]: preprocessor, {[[P6]]}, cuda-cpp-output, (device-cuda, sm_35)
// DBIN2-DAG: [[P8:[0-9]+]]: compiler, {[[P7]]}, ir, (device-cuda, sm_35)
// DBIN2-DAG: [[P9:[0-9]+]]: backend, {[[P8]]}, assembler, (device-cuda, sm_35)
// DBIN2-DAG: [[P10:[0-9]+]]: assembler, {[[P9]]}, object, (device-cuda, sm_35)
// DBIN2-DAG: [[P11:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {[[P10]]}, object

//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM2 %s
// DASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, cuda-cpp-output, (device-cuda, sm_30)
// DASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-cuda, sm_30)
// DASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-cuda, sm_30)
// DASM2-DAG: [[P4:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {[[P3]]}, assembler
// DASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// DASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, cuda-cpp-output, (device-cuda, sm_35)
// DASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-cuda, sm_35)
// DASM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-cuda, sm_35)
// DASM2-DAG: [[P9:[0-9]+]]: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {[[P8]]}, assembler
