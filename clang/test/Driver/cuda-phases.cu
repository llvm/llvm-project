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
// BIN: 0: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// BIN: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// BIN: 2: compiler, {1}, ir, (host-cuda)
// BIN: 3: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// BIN: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_30)
// BIN: 5: compiler, {4}, ir, (device-cuda, sm_30)
// BIN: 6: backend, {5}, assembler, (device-cuda, sm_30)
// BIN: 7: assembler, {6}, object, (device-cuda, sm_30)
// BIN: 8: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {7}, object
// BIN: 9: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {6}, assembler
// BIN: 10: linker, {8, 9}, cuda-fatbin, (device-cuda)
// BIN: 11: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (nvptx64-nvidia-cuda)" {10}, ir
// BIN: 12: backend, {11}, assembler, (host-cuda)
// BIN: 13: assembler, {12}, object, (host-cuda)
// BIN: 14: linker, {13}, image, (host-cuda)

//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM %s
// ASM: 0: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// ASM: 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_30)
// ASM: 2: compiler, {1}, ir, (device-cuda, sm_30)
// ASM: 3: backend, {2}, assembler, (device-cuda, sm_30)
// ASM: 4: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {3}, assembler
// ASM: 5: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// ASM: 6: preprocessor, {5}, cuda-cpp-output, (host-cuda)
// ASM: 7: compiler, {6}, ir, (host-cuda)
// ASM: 8: backend, {7}, assembler, (host-cuda)

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix=BIN2 %s
// BIN2: 0: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// BIN2: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// BIN2: 2: compiler, {1}, ir, (host-cuda)
// BIN2: 3: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// BIN2: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_30)
// BIN2: 5: compiler, {4}, ir, (device-cuda, sm_30)
// BIN2: 6: backend, {5}, assembler, (device-cuda, sm_30)
// BIN2: 7: assembler, {6}, object, (device-cuda, sm_30)
// BIN2: 8: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {7}, object
// BIN2: 9: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {6}, assembler
// BIN2: 10: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// BIN2: 11: preprocessor, {10}, cuda-cpp-output, (device-cuda, sm_35)
// BIN2: 12: compiler, {11}, ir, (device-cuda, sm_35)
// BIN2: 13: backend, {12}, assembler, (device-cuda, sm_35)
// BIN2: 14: assembler, {13}, object, (device-cuda, sm_35)
// BIN2: 15: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {14}, object
// BIN2: 16: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {13}, assembler
// BIN2: 17: linker, {8, 9, 15, 16}, cuda-fatbin, (device-cuda)
// BIN2: 18: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (nvptx64-nvidia-cuda)" {17}, ir
// BIN2: 19: backend, {18}, assembler, (host-cuda)
// BIN2: 20: assembler, {19}, object, (host-cuda)
// BIN2: 21: linker, {20}, image, (host-cuda)

//
// Test two gpu architecturess up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM2 %s
// ASM2: 0: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// ASM2: 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_30)
// ASM2: 2: compiler, {1}, ir, (device-cuda, sm_30)
// ASM2: 3: backend, {2}, assembler, (device-cuda, sm_30)
// ASM2: 4: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {3}, assembler
// ASM2: 5: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// ASM2: 6: preprocessor, {5}, cuda-cpp-output, (device-cuda, sm_35)
// ASM2: 7: compiler, {6}, ir, (device-cuda, sm_35)
// ASM2: 8: backend, {7}, assembler, (device-cuda, sm_35)
// ASM2: 9: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {8}, assembler
// ASM2: 10: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// ASM2: 11: preprocessor, {10}, cuda-cpp-output, (host-cuda)
// ASM2: 12: compiler, {11}, ir, (host-cuda)
// ASM2: 13: backend, {12}, assembler, (host-cuda)

//
// Test single gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN %s
// HBIN: 0: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HBIN: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// HBIN: 2: compiler, {1}, ir, (host-cuda)
// HBIN: 3: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, ir
// HBIN: 4: backend, {3}, assembler, (host-cuda)
// HBIN: 5: assembler, {4}, object, (host-cuda)
// HBIN: 6: linker, {5}, image, (host-cuda)

//
// Test single gpu architecture up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM %s
// HASM: 0: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HASM: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// HASM: 2: compiler, {1}, ir, (host-cuda)
// HASM: 3: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, ir
// HASM: 4: backend, {3}, assembler, (host-cuda)

//
// Test two gpu architectures with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN2 %s
// HBIN2: 0: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HBIN2: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// HBIN2: 2: compiler, {1}, ir, (host-cuda)
// HBIN2: 3: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, ir
// HBIN2: 4: backend, {3}, assembler, (host-cuda)
// HBIN2: 5: assembler, {4}, object, (host-cuda)
// HBIN2: 6: linker, {5}, image, (host-cuda)

//
// Test two gpu architectures up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM2 %s
// HASM2: 0: input, "{{.*}}cuda-phases.cu", cuda, (host-cuda)
// HASM2: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// HASM2: 2: compiler, {1}, ir, (host-cuda)
// HASM2: 3: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, ir
// HASM2: 4: backend, {3}, assembler, (host-cuda)

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN %s
// DBIN: 0: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DBIN: 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_30)
// DBIN: 2: compiler, {1}, ir, (device-cuda, sm_30)
// DBIN: 3: backend, {2}, assembler, (device-cuda, sm_30)
// DBIN: 4: assembler, {3}, object, (device-cuda, sm_30)
// DBIN: 5: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {4}, object

//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM %s
// DASM: 0: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DASM: 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_30)
// DASM: 2: compiler, {1}, ir, (device-cuda, sm_30)
// DASM: 3: backend, {2}, assembler, (device-cuda, sm_30)
// DASM: 4: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {3}, assembler

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN2 %s
// DBIN2: 0: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DBIN2: 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_30)
// DBIN2: 2: compiler, {1}, ir, (device-cuda, sm_30)
// DBIN2: 3: backend, {2}, assembler, (device-cuda, sm_30)
// DBIN2: 4: assembler, {3}, object, (device-cuda, sm_30)
// DBIN2: 5: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {4}, object
// DBIN2: 6: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// DBIN2: 7: preprocessor, {6}, cuda-cpp-output, (device-cuda, sm_35)
// DBIN2: 8: compiler, {7}, ir, (device-cuda, sm_35)
// DBIN2: 9: backend, {8}, assembler, (device-cuda, sm_35)
// DBIN2: 10: assembler, {9}, object, (device-cuda, sm_35)
// DBIN2: 11: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {10}, object

//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM2 %s
// DASM2: 0: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_30)
// DASM2: 1: preprocessor, {0}, cuda-cpp-output, (device-cuda, sm_30)
// DASM2: 2: compiler, {1}, ir, (device-cuda, sm_30)
// DASM2: 3: backend, {2}, assembler, (device-cuda, sm_30)
// DASM2: 4: offload, "device-cuda (nvptx64-nvidia-cuda:sm_30)" {3}, assembler
// DASM2: 5: input, "{{.*}}cuda-phases.cu", cuda, (device-cuda, sm_35)
// DASM2: 6: preprocessor, {5}, cuda-cpp-output, (device-cuda, sm_35)
// DASM2: 7: compiler, {6}, ir, (device-cuda, sm_35)
// DASM2: 8: backend, {7}, assembler, (device-cuda, sm_35)
// DASM2: 9: offload, "device-cuda (nvptx64-nvidia-cuda:sm_35)" {8}, assembler
