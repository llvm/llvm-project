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

/// Check for option incompatibility with -fsycl
// RUN: not %clang -### -fsycl -ffreestanding %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-ffreestanding
// RUN: not %clang -### -fsycl --offload-new-driver -static-libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-static-libstdc++
// CHK-INCOMPATIBILITY: error: invalid argument '[[INCOMPATOPT]]' not allowed with '-fsycl'
