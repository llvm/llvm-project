// Tests the phases generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.

// Test single gpu architecture with complete compilation.
//
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,NEW,NEWN %s
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 -flto -c %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,NEW,NEWLTO %s
//
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 -fgpu-rdc %s 2>&1 \
// RUN: | FileCheck -check-prefixes=BIN,NEW,NEWR %s
//
// BIN-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (host-[[T]])
// BIN-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (host-[[T]])
// BIN-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (host-[[T]])

// BIN-DAG: [[P3:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T]], (device-[[T]], [[ARCH:gfx803]])
// BIN-DAG: [[P4:[0-9]+]]: preprocessor, {[[P3]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// BIN-DAG: [[P5:[0-9]+]]: compiler, {[[P4]]}, ir, (device-[[T]], [[ARCH]])
// NEW-DAG: [[P6:[0-9]+]]: backend, {[[P5]]}, lto-bc, (device-[[T]], [[ARCH]])
// NEW-DAG: [[P7:[0-9]+]]: offload, "device-[[T]] (amdgpu-amd-amdhsa:[[ARCH]])" {[[P6]]}, lto-bc
// NEW-DAG: [[P8:[0-9]+]]: llvm-offload-binary, {[[P7]]}, image, (device-[[T]])
// NEWN-DAG: [[P9:[0-9]+]]: clang-linker-wrapper, {[[P8]]}, hip-fatbin, (device-[[T]])
// NEWLTO-DAG: [[P9:[0-9]+]]: clang-linker-wrapper, {[[P8]]}, hip-fatbin, (device-[[T]])

// NEWN-DAG: [[P10:[0-9]+]]: offload, "host-[[T]] (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-[[T]] (amdgpu-amd-amdhsa)" {[[P9]]}, ir
// NEWLTO-DAG: [[P10:[0-9]+]]: offload, "host-[[T]] (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-[[T]] (amdgpu-amd-amdhsa)" {[[P9]]}, ir
// NEWR-DAG: [[P9:[0-9]+]]: offload, "host-[[T]] (x86_64-unknown-linux-gnu)" {[[P2]]}, "device-[[T]] (x86_64-unknown-linux-gnu)" {[[P8]]}, ir
// NEWN-DAG: [[P11:[0-9]+]]: backend, {[[P10]]}, assembler, (host-[[T]])
// NEWN-DAG: [[P12:[0-9]+]]: assembler, {[[P11]]}, object, (host-[[T]])
// NEWN-DAG: [[P13:[0-9]+]]: clang-linker-wrapper, {[[P12]]}, image, (host-[[T]])
// NEWLTO-DAG: [[P11:[0-9]+]]: backend, {[[P10]]}, lto-bc, (host-hip)
// NEWR-DAG: [[P10:[0-9]+]]: backend, {[[P9]]}, assembler, (host-[[T]])
// NEWR-DAG: [[P11:[0-9]+]]: assembler, {[[P10]]}, object, (host-[[T]])
// NEWR-DAG: [[P12:[0-9]+]]: clang-linker-wrapper, {[[P11]]}, image, (host-[[T]])

//
// Test single gpu architecture with compile to relocatable in device-only
// compilation mode.
//
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only -fhip-emit-relocatable 2>&1 \
// RUN: | FileCheck -check-prefixes=RELOC %s
//
// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -ccc-print-phases \
// RUN: --cuda-gpu-arch=gfx803 %s --cuda-device-only -fhip-emit-relocatable -Wl,--disable-new-dtags \
// RUN: 2>&1 | FileCheck -check-prefixes=RELOC %s
//
// RELOC-DAG: [[P0:[0-9]+]]: input, "{{.*}}hip-phases.hip", [[T:hip]], (device-[[T]], [[ARCH:gfx803]])
// RELOC-DAG: [[P1:[0-9]+]]: preprocessor, {[[P0]]}, [[T]]-cpp-output, (device-[[T]], [[ARCH]])
// RELOC-DAG: [[P2:[0-9]+]]: compiler, {[[P1]]}, ir, (device-[[T]], [[ARCH]])
// RELOC-DAG: [[P3:[0-9]+]]: backend, {[[P2]]}, assembler, (device-[[T]], [[ARCH]])
// RELOC-DAG: [[P4:[0-9]+]]: assembler, {[[P3]]}, object, (device-[[T]], [[ARCH]])
// RELOC-NOT: linker
// RELOC-DAG: [[P5:[0-9]+]]: offload, "device-[[T]] (amdgpu-amd-amdhsa:[[ARCH]])" {[[P4]]}
// RELOC-NOT: host

//
// Test the bindings in LTO-mode.
//
// RUN: %clang -### --target=x86_64-linux-gnu -ccc-print-phases \
// RUN:        --offload-arch=gfx90a --offload-arch=gfx908 -foffload-lto -fgpu-rdc -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=LTO %s
//      LTO: 0: input, "[[INPUT:.+]]", hip, (host-hip)
// LTO-NEXT: 1: preprocessor, {0}, hip-cpp-output, (host-hip)
// LTO-NEXT: 2: compiler, {1}, ir, (host-hip)
// LTO-NEXT: 3: input, "[[INPUT]]", hip, (device-hip, gfx908)
// LTO-NEXT: 4: preprocessor, {3}, hip-cpp-output, (device-hip, gfx908)
// LTO-NEXT: 5: compiler, {4}, ir, (device-hip, gfx908)
// LTO-NEXT: 6: backend, {5}, lto-bc, (device-hip, gfx908)
// LTO-NEXT: 7: offload, "device-hip (amdgpu-amd-amdhsa:gfx908)" {6}, lto-bc
// LTO-NEXT: 8: input, "[[INPUT]]", hip, (device-hip, gfx90a)
// LTO-NEXT: 9: preprocessor, {8}, hip-cpp-output, (device-hip, gfx90a)
// LTO-NEXT: 10: compiler, {9}, ir, (device-hip, gfx90a)
// LTO-NEXT: 11: backend, {10}, lto-bc, (device-hip, gfx90a)
// LTO-NEXT: 12: offload, "device-hip (amdgpu-amd-amdhsa:gfx90a)" {11}, lto-bc
// LTO-NEXT: 13: llvm-offload-binary, {7, 12}, image, (device-hip)
// LTO-NEXT: 14: offload, "host-hip (x86_64-unknown-linux-gnu)" {2}, "device-hip (x86_64-unknown-linux-gnu)" {13}, ir
// LTO-NEXT: 15: backend, {14}, assembler, (host-hip)
// LTO-NEXT: 16: assembler, {15}, object, (host-hip)

//
// Test device-only compilation when not bundling.
//
// RUN: %clang -### --target=x86_64-linux-gnu -ccc-print-phases \
// RUN:        --offload-device-only --offload-arch=gfx90a -emit-llvm -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEVICE-ONLY %s
//      DEVICE-ONLY: 0: input, "[[INPUT:.+]]", hip, (device-hip, gfx90a)
// DEVICE-ONLY-NEXT: 1: preprocessor, {0}, hip-cpp-output, (device-hip, gfx90a)
// DEVICE-ONLY-NEXT: 2: compiler, {1}, ir, (device-hip, gfx90a)
// DEVICE-ONLY-NEXT: 3: backend, {2}, ir, (device-hip, gfx90a)
// DEVICE-ONLY-NEXT: 4: offload, "device-hip (amdgpu-amd-amdhsa:gfx90a)" {3}, none

//
// Test bundling SPIR-V targets.
//
// RUN: %clang -### --target=x86_64-linux-gnu -ccc-print-phases \
// RUN:        --offload-device-only -no-use-spirv-backend \
// RUN:        --offload-arch=amdgcnspirv,gfx1030 %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPIRV-ONLY %s
//      SPIRV-ONLY: 0: input, "[[INPUT:.+]]", hip, (device-hip, gfx1030)
// SPIRV-ONLY-NEXT: 1: preprocessor, {0}, hip-cpp-output, (device-hip, gfx1030)
// SPIRV-ONLY-NEXT: 2: compiler, {1}, ir, (device-hip, gfx1030)
// SPIRV-ONLY-NEXT: 3: backend, {2}, assembler, (device-hip, gfx1030)
// SPIRV-ONLY-NEXT: 4: assembler, {3}, object, (device-hip, gfx1030)
// SPIRV-ONLY-NEXT: 5: linker, {4}, image, (device-hip, gfx1030)
// SPIRV-ONLY-NEXT: 6: offload, "device-hip (amdgpu-amd-amdhsa:gfx1030)" {5}, image
// SPIRV-ONLY-NEXT: 7: input, "[[INPUT]]", hip, (device-hip, amdgcnspirv)
// SPIRV-ONLY-NEXT: 8: preprocessor, {7}, hip-cpp-output, (device-hip, amdgcnspirv)
// SPIRV-ONLY-NEXT: 9: compiler, {8}, ir, (device-hip, amdgcnspirv)
// SPIRV-ONLY-NEXT: 10: backend, {9}, lto-bc, (device-hip, amdgcnspirv)
// SPIRV-ONLY-NEXT: 11: linker, {10}, image, (device-hip, amdgcnspirv)
// SPIRV-ONLY-NEXT: 12: offload, "device-hip (spirv64-amd-amdhsa:amdgcnspirv)" {11}, image
// SPIRV-ONLY-NEXT: 13: linker, {6, 12}, hip-fatbin, (device-hip)
// SPIRV-ONLY-NEXT: 14: offload, "device-hip (amdgpu-amd-amdhsa)" {13}, none
