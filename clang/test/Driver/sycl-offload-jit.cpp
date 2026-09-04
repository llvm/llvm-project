/// Perform several driver tests for SYCL offloading for JIT

/// Check the phases graph with -fsycl. Use of -fsycl enables offload
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES %s
// RUN: %clang_cl -ccc-print-phases --target=x86_64-pc-windows-msvc -fsycl -- %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES %s
// CHK-PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-PHASES-NEXT: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NEXT: 2: compiler, {1}, ir, (host-sycl)
// CHK-PHASES-NEXT: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHK-PHASES-NEXT: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-NEXT: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-NEXT: 6: backend, {5}, ir, (device-sycl)
// CHK-PHASES-NEXT: 7: offload, "device-sycl (spirv64-unknown-unknown)" {6}, ir
// CHK-PHASES-NEXT: 8: llvm-offload-binary, {7}, image, (device-sycl)
// CHK-PHASES-NEXT: 9: offload, "host-sycl (x86_64{{.*}})" {2}, "device-sycl (x86_64{{.*}})" {8}, ir
// CHK-PHASES-NEXT: 10: backend, {9}, assembler, (host-sycl)
// CHK-PHASES-NEXT: 11: assembler, {10}, object, (host-sycl)
// CHK-PHASES-NEXT: 12: clang-linker-wrapper, {11}, image, (host-sycl)

/// Check expected default values for device compilation when using -fsycl as
/// well as llvm-offload-binary inputs.
// RUN: %clang -### -fsycl -c --target=x86_64-unknown-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEVICE-TRIPLE %s
// CHK-DEVICE-TRIPLE: "-cc1"{{.*}} "-triple" "spirv64-unknown-unknown"
// CHK-DEVICE-TRIPLE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHK-DEVICE-TRIPLE-SAME: "-fsycl-is-device"
// CHK-DEVICE-TRIPLE-SAME: "-O2"
// CHK-DEVICE-TRIPLE: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spirv64-unknown-unknown,arch=generic,kind=sycl"

// Check that -fsycl -fno-sycl does not pass libLLVMSYCL.so to the linker.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-sycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SYCL-RT %s
// CHECK-NO-SYCL-RT-NOT: libLLVMSYCL.so

/// Check -fsycl-is-device is passed when compiling for the device.
/// Check -fsycl-is-host is passed when compiling for host.
// RUN: %clang -### -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYCL-IS-DEVICE,CHK-FSYCL-IS-HOST %s
// RUN: %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-DEVICE %s
// RUN: %clang_cl -### -fsycl -c -- %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-FSYCL-IS-DEVICE,CHK-FSYCL-IS-HOST %s
// RUN: %clang -### -fsycl -fsycl-host-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-FSYCL-IS-HOST %s
// CHK-FSYCL-IS-DEVICE: "-cc1"{{.*}} "-fsycl-is-device" {{.*}} "-emit-llvm-bc"
// CHK-FSYCL-IS-HOST: "-cc1"{{.*}} "-fsycl-is-host"

/// Check that SYCL compilation defaults to relocatable device code (-fgpu-rdc
/// is passed to both the device and the host -cc1 invocation) and that
/// -fno-gpu-rdc disables it.
// RUN: %clang -### -fsycl -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SYCL-RDC,CHK-SYCL-RDC-HOST %s
// RUN: %clang -### -fsycl -fgpu-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-SYCL-RDC,CHK-SYCL-RDC-HOST %s
// RUN: %clang -### -fsycl -fno-gpu-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SYCL-NORDC %s
// CHK-SYCL-RDC: "-cc1"{{.*}} "-fsycl-is-device" {{.*}} "-fgpu-rdc"
// CHK-SYCL-RDC-HOST: "-cc1"{{.*}} "-fsycl-is-host" {{.*}} "-fgpu-rdc"
// CHK-SYCL-NORDC-NOT: "-fgpu-rdc"

/// Check the phases graph in non-RDC mode.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=CHK-PHASES-NORDC %s
// CHK-PHASES-NORDC: 6: backend, {5}, ir, (device-sycl)
// CHK-PHASES-NORDC-NEXT: 7: offload, "device-sycl (spirv64-unknown-unknown)" {6}, ir
// CHK-PHASES-NORDC-NEXT: 8: llvm-offload-binary, {7}, image, (device-sycl)
// CHK-PHASES-NORDC-NEXT: 9: clang-linker-wrapper, {8}, sycl-fatbin, (device-sycl)
// CHK-PHASES-NORDC-NEXT: 10: offload, "host-sycl (x86_64{{.*}})" {2}, "device-sycl (spirv64{{.*}})" {9}, ir
// CHK-PHASES-NORDC-NEXT: 11: backend, {10}, assembler, (host-sycl)
// CHK-PHASES-NORDC-NEXT: 12: assembler, {11}, object, (host-sycl)
// CHK-PHASES-NORDC-NEXT: 13: clang-linker-wrapper, {12}, image, (host-sycl)

/// With multiple architectures the packaged binary holds an entry per
/// architecture, and a single fat binary is expected to reach the host.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=generic --offload-arch=bmg_g21 \
// RUN:   -c %s 2>&1 | FileCheck -check-prefixes=CHK-PHASES-NORDC-ARCHS %s
// CHK-PHASES-NORDC-ARCHS: 7: offload, "device-sycl (spirv64-unknown-unknown:bmg_g21)" {6}, ir
// CHK-PHASES-NORDC-ARCHS: 12: offload, "device-sycl (spirv64-unknown-unknown:generic)" {11}, ir
// CHK-PHASES-NORDC-ARCHS-NEXT: 13: llvm-offload-binary, {7, 12}, image, (device-sycl)
// CHK-PHASES-NORDC-ARCHS-NEXT: 14: clang-linker-wrapper, {13}, sycl-fatbin, (device-sycl)

/// Multiple device triples are not supported today in non-RDC mode.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   --offload-targets=spirv64-unknown-unknown,spirv32-unknown-unknown -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-MULTI-TRIPLE %s
// CHK-NORDC-MULTI-TRIPLE: error: '-fno-gpu-rdc' is not supported with multiple SYCL offloading targets

/// Multiple device triples are supported in RDC mode.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fgpu-rdc \
// RUN:   --offload-targets=spirv64-unknown-unknown,spirv32-unknown-unknown -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RDC-MULTI-TRIPLE %s
// CHK-RDC-MULTI-TRIPLE-NOT: error:
// CHK-RDC-MULTI-TRIPLE: "-cc1" "-triple" "spirv32-unknown-unknown"{{.*}} "-fsycl-is-device"
// CHK-RDC-MULTI-TRIPLE: "-cc1" "-triple" "spirv64-unknown-unknown"{{.*}} "-fsycl-is-device"

/// A single target repeated is one target.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   --offload-targets=spirv64-unknown-unknown,spirv64-unknown-unknown -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-DUP-TRIPLE %s
// CHK-NORDC-DUP-TRIPLE-NOT: error:
// CHK-NORDC-DUP-TRIPLE: clang-linker-wrapper{{.*}} "--emit-fatbin-only"

/// Check that in non-RDC mode clang-linker-wrapper finalizes the packaged
/// device images into a fat binary rather than a host object, and that the
/// binary is included into the host compilation via -foffload-include-binary.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-INCLUDE %s \
// RUN:     --implicit-check-not='"-fembed-offload-object='
// CHK-NORDC-INCLUDE: clang-linker-wrapper{{.*}} "--linker-path={{.*}}clang-sycl-linker" "--emit-fatbin-only" "-o" "[[FB:.*]].syclfb"
// CHK-NORDC-INCLUDE: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-foffload-include-binary" "[[FB]].syclfb"

/// Conversely, RDC mode embeds unlinked device code via -fembed-offload-object.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fgpu-rdc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-RDC-EMBED %s \
// RUN:     --implicit-check-not='"-foffload-include-binary"' \
// RUN:     --implicit-check-not='"--emit-fatbin-only"'
// CHK-RDC-EMBED: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-fembed-offload-object=

/// -v reaches the per-TU device finalize and the wrapper itself.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc -v -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NORDC-VERBOSE %s
// CHK-NORDC-VERBOSE: clang-linker-wrapper{{.*}} "--device-compiler=spirv64-unknown-unknown=-v"
// CHK-NORDC-VERBOSE-SAME: "--wrapper-verbose"
// CHK-NORDC-VERBOSE-SAME: "--emit-fatbin-only"

/// -flto on a SYCL command line requests *host* LTO. It must not divert the
/// per-TU device finalize to llvm-lto, which would write bitcode where a
/// finalized device image is expected; the device link is unaffected and the
/// binary is still included at compile time.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   -flto -c %s 2>&1 | FileCheck -check-prefix=CHK-NORDC-LTO %s \
// RUN:     --implicit-check-not=llvm-lto
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   -flto %s 2>&1 | FileCheck -check-prefix=CHK-NORDC-LTO %s \
// RUN:     --implicit-check-not=llvm-lto
// CHK-NORDC-LTO: clang-linker-wrapper{{.*}} "--linker-path={{.*}}clang-sycl-linker" "--emit-fatbin-only"
// CHK-NORDC-LTO: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-foffload-include-binary"

// Check that --allow-partial-linkage and --create-library are not passed to
// clang-linker-wrapper for SYCL (they are spirv-link flags, not clang-sycl-linker flags).
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NO-SPIRVLINK-FLAGS %s
// CHECK-NO-SPIRVLINK-FLAGS-NOT: --device-linker=spirv64-unknown-unknown=--allow-partial-linkage
// CHECK-NO-SPIRVLINK-FLAGS-NOT: --device-linker=spirv64-unknown-unknown=--create-library

/// Check -fsycl-device-image-split= is forwarded to clang-sycl-linker as the
/// corresponding --module-split-mode= value.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-image-split=kernel %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-KERNEL %s
// CHK-SPLIT-KERNEL: clang-linker-wrapper{{.*}}"--device-linker=spirv64-unknown-unknown=--module-split-mode=kernel"
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-image-split=translation_unit %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-TU %s
// CHK-SPLIT-TU: clang-linker-wrapper{{.*}}"--device-linker=spirv64-unknown-unknown=--module-split-mode=translation_unit"
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-image-split=link_unit %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-LU %s
// CHK-SPLIT-LU: clang-linker-wrapper{{.*}}"--device-linker=spirv64-unknown-unknown=--module-split-mode=link_unit"

/// Check the bare -fsycl-device-image-split flag aliases to 'translation_unit'.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-image-split %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-TU %s

/// Check that without -fsycl-device-image-split, no --module-split-mode= is passed.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-SPLIT %s
// CHK-NO-SPLIT-NOT: --module-split-mode=

/// Check an invalid -fsycl-device-image-split= value is diagnosed.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-image-split=bogus %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-INVALID %s
// CHK-SPLIT-INVALID: error: invalid value 'bogus' in '-fsycl-device-image-split='

/// Check -fsycl-device-image-split is unused when not linking (e.g. -c).
// RUN: %clang -### -c --target=x86_64-unknown-linux-gnu -fsycl -fsycl-device-image-split=kernel %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-UNUSED %s
// CHK-SPLIT-UNUSED: warning: argument unused during compilation: '-fsycl-device-image-split=kernel'

/// In non-RDC mode the split does happen while compiling.
// RUN: %clang -### -c --target=x86_64-unknown-linux-gnu -fsycl -fno-gpu-rdc \
// RUN:   -fsycl-device-image-split=kernel %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-SPLIT-KERNEL %s \
// RUN:     --implicit-check-not='argument unused during compilation'

/// Check for option incompatibility with -fsycl
// RUN: not %clang -### -fsycl -ffreestanding %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-ffreestanding
// RUN: not %clang -### -fsycl -static-libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-static-libstdc++
// CHK-INCOMPATIBILITY: error: invalid argument '[[INCOMPATOPT]]' not allowed with '-fsycl'
