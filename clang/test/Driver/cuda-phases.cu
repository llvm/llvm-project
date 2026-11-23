// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

//
// Test the phases generated for CUDA offloading.
//
// RUN: %clang -### --target=powerpc64le-ibm-linux-gnu -ccc-print-phases -fgpu-rdc \
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
// NEW-DRIVER-RDC-NEXT: 15: llvm-offload-binary, {8, 14}, image, (device-cuda)
// NEW-DRIVER-RDC-NEXT: 16: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (powerpc64le-ibm-linux-gnu)" {15}, ir
// NEW-DRIVER-RDC-NEXT: 17: backend, {16}, assembler, (host-cuda)
// NEW-DRIVER-RDC-NEXT: 18: assembler, {17}, object, (host-cuda)
// NEW-DRIVER-RDC-NEXT: 19: clang-linker-wrapper, {18}, image, (host-cuda)

// RUN: %clang -### -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
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

// RUN: %clang -### --target=powerpc64le-ibm-linux-gnu -ccc-print-phases \
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
// Test the phases in LTO-mode.
//
// RUN: %clang -### -target powerpc64le-ibm-linux-gnu -ccc-print-phases \
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
// LTO-NEXT: 13: llvm-offload-binary, {7, 12}, image, (device-cuda)
// LTO-NEXT: 14: offload, "host-cuda (powerpc64le-ibm-linux-gnu)" {2}, "device-cuda (powerpc64le-ibm-linux-gnu)" {13}, ir
// LTO-NEXT: 15: backend, {14}, assembler, (host-cuda)
// LTO-NEXT: 16: assembler, {15}, object, (host-cuda)

//
// Test that invalid architectures do not create actions.
//
// RUN: not %clang -### --target=powerpc64le-ibm-linux-gnu \
// RUN:        -ccc-print-phases --offload-arch=sm_999 -fgpu-rdc -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=INVALID-ARCH %s
//      INVALID-ARCH: error: unsupported CUDA gpu architecture: sm_999
//      INVALID-ARCH: 0: input, "[[INPUT:.+]]", cuda
// INVALID-ARCH-NEXT: 1: preprocessor, {0}, cuda-cpp-output
// INVALID-ARCH-NEXT: 2: compiler, {1}, ir
// INVALID-ARCH-NEXT: 3: backend, {2}, assembler
// INVALID-ARCH-NEXT: 4: assembler, {3}, object
