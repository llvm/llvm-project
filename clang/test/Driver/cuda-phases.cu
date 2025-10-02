// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

// Test single gpu architecture with complete compilation.
//
// Test CUDA NVPTX phases.
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN %s
//
// BIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// BIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// BIN-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH:sm_30]])
// BIN-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P8:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P7]]}, object
// BIN-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH]])" {[[P6]]}, assembler
// BIN-DAG: [[P10:[0-9]+]]: linker, {[[P8]], [[P9]]}, cuda-fatbin, (device-[[T]])
// BIN-DAG: [[P11:[0-9]+]]: offload, "host-[[T]] (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-[[T]] ([[TRIPLE]])" {[[P10]]}, ir
// BIN-DAG: [[P12:[0-9]+]]: backend, {[[P11]]}, assembler, (host-[[T]])
// BIN-DAG: [[P13:[0-9]+]]: assembler, {[[P12]]}, object, (host-[[T]])
// BIN-DAG: [[P14:[0-9]+]]: linker, {[[P13]]}, image, (host-[[T]])

//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM %s
// ASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// ASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// ASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// ASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// ASM-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P3]]}, assembler
// ASM-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (host-[[T]])
// ASM-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (host-[[T]])
// ASM-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (host-[[T]])
// ASM-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (host-[[T]])

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN2 %s
// BIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// BIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// BIN2-DAG: [[P3:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH1:sm_30]])
// BIN2-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, assembler, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P7:[0-9]+]]: assembler, {[[P6]]}, object, (device-[[T]], [[ARCH1]])
// BIN2-DAG: [[P8:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH1]])" {[[P7]]}, object
// BIN2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH1]])" {[[P6]]}, assembler
// BIN2-DAG: [[P10:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35]])
// BIN2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P14:[0-9]+]]: assembler, {[[P13]]}, object, (device-[[T]], [[ARCH2]])
// BIN2-DAG: [[P15:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P14]]}, object
// BIN2-DAG: [[P16:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P13]]}, assembler
// BIN2-DAG: [[P17:[0-9]+]]: linker, {[[P8]], [[P9]], [[P15]], [[P16]]}, cuda-fatbin, (device-[[T]])
// BIN2-DAG: [[P18:[0-9]+]]: offload, "host-[[T]] (powerpc64le-ibm-linux-gnu)" {[[P2]]}, "device-[[T]] ([[TRIPLE]])" {[[P17]]}, ir
// BIN2-DAG: [[P19:[0-9]+]]: backend, {[[P18]]}, assembler, (host-[[T]])
// BIN2-DAG: [[P20:[0-9]+]]: assembler, {[[P19]]}, object, (host-[[T]])
// BIN2-DAG: [[P21:[0-9]+]]: linker, {[[P20]]}, image, (host-[[T]])

//
// Test two gpu architecturess up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -S 2>&1 \
// RUN: | FileCheck -check-prefixes=ASM2 %s
// ASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH1:sm_30]])
// ASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH1]])
// ASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH1]])
// ASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH1]])
// ASM2-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH1]])" {[[P3]]}, assembler
// ASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35]])
// ASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// ASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// ASM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-[[T]], [[ARCH2]])
// ASM2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P8]]}, assembler
// ASM2-DAG: [[P10:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (host-[[T]])
// ASM2-DAG: [[P11:[0-9]+]]: preprocessor, {[[P10]]}, [[T]]-cpp-output, (host-[[T]])
// ASM2-DAG: [[P12:[0-9]+]]: compiler, {[[P11]]}, ir, (host-[[T]])
// ASM2-DAG: [[P13:[0-9]+]]: backend, {[[P12]]}, assembler, (host-[[T]])

//
// Test single gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN %s
// HBIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
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
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=HASM %s
// HASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// HASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HASM-NOT: device

//
// Test two gpu architectures with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefixes=HBIN2 %s
// HBIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
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
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only -S \
// RUN: 2>&1 | FileCheck -check-prefixes=HASM2 %s
// HASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (host-[[T]])
// HASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// HASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])
// HASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (host-[[T]])
// HASM2-NOT: device

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefixes=DBIN %s
// DBIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DBIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// DBIN-DAG: [[P5:[0-9]+]]: offload, "device-[[T]] (nvptx64-nvidia-cuda:[[ARCH]])" {[[P4]]}, object
// DBIN-NOT: host
//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefixes=DASM %s
// DASM-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DASM-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DASM-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P3]]}, assembler
// DASM-NOT: host

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefixes=DBIN2 %s
// DBIN2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DBIN2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// DBIN2-DAG: [[P5:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P4]]}, object
// DBIN2-DAG: [[P6:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35]])
// DBIN2-DAG: [[P7:[0-9]+]]: preprocessor, {[[P6]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P8:[0-9]+]]: compiler, {[[P7]]}, ir, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P9:[0-9]+]]: backend, {[[P8]]}, assembler, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P10:[0-9]+]]: assembler, {[[P9]]}, object, (device-[[T]], [[ARCH2]])
// DBIN2-DAG: [[P11:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P10]]}, object
// DBIN2-NOT: host
//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
// RUN:   --no-offload-new-driver --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only -S \
// RUN: 2>&1 | FileCheck -check-prefixes=DASM2 %s
// DASM2-DAG: [[P0:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T:cuda]], (device-[[T]], [[ARCH:sm_30]])
// DASM2-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// DASM2-DAG: [[P4:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE:nvptx64-nvidia-cuda]]:[[ARCH]])" {[[P3]]}, assembler
// DASM2-DAG: [[P5:[0-9]+]]: input, "{{.*}}cuda-phases.cu", [[T]], (device-[[T]], [[ARCH2:sm_35]])
// DASM2-DAG: [[P6:[0-9]+]]: preprocessor, {[[P5]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P7:[0-9]+]]: compiler, {[[P6]]}, ir, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P8:[0-9]+]]: backend, {[[P7]]}, assembler, (device-[[T]], [[ARCH2]])
// DASM2-DAG: [[P9:[0-9]+]]: offload, "device-[[T]] ([[TRIPLE]]:[[ARCH2]])" {[[P8]]}, assembler
// DASM2-NOT: host

//
// Test the phases generated when using the new offloading driver.
//
// RUN: %clang -### --target=powerpc64le-ibm-linux-gnu -ccc-print-phases --offload-new-driver -fgpu-rdc \
// RUN:   --offload-arch=sm_52 --offload-arch=sm_70 %s 2>&1 | FileCheck --check-prefix=NEW-DRIVER-RDC %s
//      NEW-DRIVER-RDC: 0: input, "[[INPUT:.+]]", cuda
// NEW-DRIVER-RDC-NEXT: 1: preprocessor, {0}, cuda-cpp-output
// NEW-DRIVER-RDC-NEXT: 2: compiler, {1}, ir
// NEW-DRIVER-RDC-NEXT: 3: input, "[[INPUT]]", cuda, (device-cuda, sm_52)
// NEW-DRIVER-RDC-NEXT: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_52)
// NEW-DRIVER-RDC-NEXT: 5: compiler, {4}, ir, (device-cuda, sm_52)
// NEW-DRIVER-RDC-NEXT: 6: backend, {5}, assembler, (device-cuda, sm_52)
// NEW-DRIVER-RDC-NEXT: 7: assembler, {6}, object, (device-cuda, sm_52)
// NEW-DRIVER-RDC-NEXT: 8: offload, "device-cuda (nvptx64-nvidia-cuda:sm_52)" {7}, object
// NEW-DRIVER-RDC-NEXT: 9: input, "[[INPUT]]", cuda, (device-cuda, sm_70)
// NEW-DRIVER-RDC-NEXT: 10: preprocessor, {9}, cuda-cpp-output, (device-cuda, sm_70)
// NEW-DRIVER-RDC-NEXT: 11: compiler, {10}, ir, (device-cuda, sm_70)
// NEW-DRIVER-RDC-NEXT: 12: backend, {11}, assembler, (device-cuda, sm_70)
// NEW-DRIVER-RDC-NEXT: 13: assembler, {12}, object, (device-cuda, sm_70)
// NEW-DRIVER-RDC-NEXT: 14: offload, "device-cuda (nvptx64-nvidia-cuda:sm_70)" {13}, object
// NEW-DRIVER-RDC-NEXT: 15: clang-offload-packager, {8, 14}, image, (device-cuda)
// NEW-DRIVER-RDC-NEXT: 16: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (powerpc64le-ibm-linux-gnu)" {15}, ir
// NEW-DRIVER-RDC-NEXT: 17: backend, {16}, assembler, (host-cuda)
// NEW-DRIVER-RDC-NEXT: 18: assembler, {17}, object, (host-cuda)
// NEW-DRIVER-RDC-NEXT: 19: clang-linker-wrapper, {18}, image, (host-cuda)

// RUN: %clang -### -target powerpc64le-ibm-linux-gnu -ccc-print-phases --offload-new-driver \
// RUN:   --offload-arch=sm_52 --offload-arch=sm_70 %s 2>&1 | FileCheck --check-prefix=NEW-DRIVER %s
//      NEW-DRIVER: 0: input, "[[CUDA:.+]]", cuda, (host-cuda)
// NEW-DRIVER-NEXT: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// NEW-DRIVER-NEXT: 2: compiler, {1}, ir, (host-cuda)
// NEW-DRIVER-NEXT: 3: input, "[[CUDA]]", cuda, (device-cuda, sm_52)
// NEW-DRIVER-NEXT: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_52)
// NEW-DRIVER-NEXT: 5: compiler, {4}, ir, (device-cuda, sm_52)
// NEW-DRIVER-NEXT: 6: backend, {5}, assembler, (device-cuda, sm_52)
// NEW-DRIVER-NEXT: 7: assembler, {6}, object, (device-cuda, sm_52)
// NEW-DRIVER-NEXT: 8: offload, "device-cuda (nvptx64-nvidia-cuda:sm_52)" {7}, "device-cuda (nvptx64-nvidia-cuda:sm_52)" {6}, object
// NEW-DRIVER-NEXT: 9: input, "[[CUDA]]", cuda, (device-cuda, sm_70)
// NEW-DRIVER-NEXT: 10: preprocessor, {9}, cuda-cpp-output, (device-cuda, sm_70)
// NEW-DRIVER-NEXT: 11: compiler, {10}, ir, (device-cuda, sm_70)
// NEW-DRIVER-NEXT: 12: backend, {11}, assembler, (device-cuda, sm_70)
// NEW-DRIVER-NEXT: 13: assembler, {12}, object, (device-cuda, sm_70)
// NEW-DRIVER-NEXT: 14: offload, "device-cuda (nvptx64-nvidia-cuda:sm_70)" {13}, "device-cuda (nvptx64-nvidia-cuda:sm_70)" {12}, object
// NEW-DRIVER-NEXT: 15: linker, {8, 14}, cuda-fatbin, (device-cuda)
// NEW-DRIVER-NEXT: 16: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (nvptx64-nvidia-cuda)" {15}, ir
// NEW-DRIVER-NEXT: 17: backend, {16}, assembler, (host-cuda)
// NEW-DRIVER-NEXT: 18: assembler, {17}, object, (host-cuda)
// NEW-DRIVER-NEXT: 19: clang-linker-wrapper, {18}, image, (host-cuda)

// RUN: %clang -### --target=powerpc64le-ibm-linux-gnu -ccc-print-phases --offload-new-driver \
// RUN:   --offload-arch=sm_52 --offload-arch=sm_70 %s %S/Inputs/empty.cpp 2>&1 | FileCheck --check-prefix=NON-CUDA-INPUT %s

//      NON-CUDA-INPUT: 0: input, "[[CUDA:.+]]", cuda, (host-cuda)
// NON-CUDA-INPUT-NEXT: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// NON-CUDA-INPUT-NEXT: 2: compiler, {1}, ir, (host-cuda)
// NON-CUDA-INPUT-NEXT: 3: input, "[[CUDA]]", cuda, (device-cuda, sm_52)
// NON-CUDA-INPUT-NEXT: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_52)
// NON-CUDA-INPUT-NEXT: 5: compiler, {4}, ir, (device-cuda, sm_52)
// NON-CUDA-INPUT-NEXT: 6: backend, {5}, assembler, (device-cuda, sm_52)
// NON-CUDA-INPUT-NEXT: 7: assembler, {6}, object, (device-cuda, sm_52)
// NON-CUDA-INPUT-NEXT: 8: offload, "device-cuda (nvptx64-nvidia-cuda:sm_52)" {7}, "device-cuda (nvptx64-nvidia-cuda:sm_52)" {6}, object
// NON-CUDA-INPUT-NEXT: 9: input, "[[CUDA]]", cuda, (device-cuda, sm_70)
// NON-CUDA-INPUT-NEXT: 10: preprocessor, {9}, cuda-cpp-output, (device-cuda, sm_70)
// NON-CUDA-INPUT-NEXT: 11: compiler, {10}, ir, (device-cuda, sm_70)
// NON-CUDA-INPUT-NEXT: 12: backend, {11}, assembler, (device-cuda, sm_70)
// NON-CUDA-INPUT-NEXT: 13: assembler, {12}, object, (device-cuda, sm_70)
// NON-CUDA-INPUT-NEXT: 14: offload, "device-cuda (nvptx64-nvidia-cuda:sm_70)" {13}, "device-cuda (nvptx64-nvidia-cuda:sm_70)" {12}, object
// NON-CUDA-INPUT-NEXT: 15: linker, {8, 14}, cuda-fatbin, (device-cuda)
// NON-CUDA-INPUT-NEXT: 16: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (nvptx64-nvidia-cuda)" {15}, ir
// NON-CUDA-INPUT-NEXT: 17: backend, {16}, assembler, (host-cuda)
// NON-CUDA-INPUT-NEXT: 18: assembler, {17}, object, (host-cuda)
// NON-CUDA-INPUT-NEXT: 19: input, "[[CPP:.+]]", c++, (host-cuda)
// NON-CUDA-INPUT-NEXT: 20: preprocessor, {19}, c++-cpp-output, (host-cuda)
// NON-CUDA-INPUT-NEXT: 21: compiler, {20}, ir, (host-cuda)
// NON-CUDA-INPUT-NEXT: 22: backend, {21}, assembler, (host-cuda)
// NON-CUDA-INPUT-NEXT: 23: assembler, {22}, object, (host-cuda)
// NON-CUDA-INPUT-NEXT: 24: clang-linker-wrapper, {18, 23}, image, (host-cuda)

//
// Test the phases using the new driver in LTO-mode.
//
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu --offload-new-driver -ccc-print-phases \
// RUN:        --offload-arch=sm_70 --offload-arch=sm_52 -foffload-lto -fgpu-rdc -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=LTO %s
//      LTO: 0: input, "[[INPUT:.+]]", cuda, (host-cuda)
// LTO-NEXT: 1: preprocessor, {0}, cuda-cpp-output, (host-cuda)
// LTO-NEXT: 2: compiler, {1}, ir, (host-cuda)
// LTO-NEXT: 3: input, "[[INPUT]]", cuda, (device-cuda, sm_52)
// LTO-NEXT: 4: preprocessor, {3}, cuda-cpp-output, (device-cuda, sm_52)
// LTO-NEXT: 5: compiler, {4}, ir, (device-cuda, sm_52)
// LTO-NEXT: 6: backend, {5}, lto-bc, (device-cuda, sm_52)
// LTO-NEXT: 7: offload, "device-cuda (nvptx64-nvidia-cuda:sm_52)" {6}, lto-bc
// LTO-NEXT: 8: input, "[[INPUT]]", cuda, (device-cuda, sm_70)
// LTO-NEXT: 9: preprocessor, {8}, cuda-cpp-output, (device-cuda, sm_70)
// LTO-NEXT: 10: compiler, {9}, ir, (device-cuda, sm_70)
// LTO-NEXT: 11: backend, {10}, lto-bc, (device-cuda, sm_70)
// LTO-NEXT: 12: offload, "device-cuda (nvptx64-nvidia-cuda:sm_70)" {11}, lto-bc
// LTO-NEXT: 13: clang-offload-packager, {7, 12}, image, (device-cuda)
// LTO-NEXT: 14: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (powerpc64le-ibm-linux-gnu)" {13}, ir
// LTO-NEXT: 15: backend, {14}, assembler, (host-cuda)
// LTO-NEXT: 16: assembler, {15}, object, (host-cuda)

//
// Test that the new driver does not create actions for invalid architectures.
//
// RUN: not %clang -### --target=powerpc64le-ibm-linux-gnu --offload-new-driver \
// RUN:        -ccc-print-phases --offload-arch=sm_999 -fgpu-rdc -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=INVALID-ARCH %s
//      INVALID-ARCH: error: unsupported CUDA gpu architecture: sm_999
//      INVALID-ARCH: 0: input, "[[INPUT:.+]]", cuda
// INVALID-ARCH-NEXT: 1: preprocessor, {0}, cuda-cpp-output
// INVALID-ARCH-NEXT: 2: compiler, {1}, ir
// INVALID-ARCH-NEXT: 3: backend, {2}, assembler
// INVALID-ARCH-NEXT: 4: assembler, {3}, object
