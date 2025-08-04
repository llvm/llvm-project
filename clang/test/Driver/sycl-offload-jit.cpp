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
// CHK-PHASES-NEXT: 8: clang-offload-packager, {7}, image, (device-sycl)
// CHK-PHASES-NEXT: 9: offload, "host-sycl (x86_64{{.*}})" {2}, "device-sycl (x86_64{{.*}})" {8}, ir
// CHK-PHASES-NEXT: 10: backend, {9}, assembler, (host-sycl)
// CHK-PHASES-NEXT: 11: assembler, {10}, object, (host-sycl)
// CHK-PHASES-NEXT: 12: clang-linker-wrapper, {11}, image, (host-sycl)

/// Check expected default values for device compilation when using -fsycl as
/// well as clang-offload-packager inputs.
// RUN: %clang -### -fsycl -c --target=x86_64-unknown-linux-gnu %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DEVICE-TRIPLE %s
// CHK-DEVICE-TRIPLE: "-cc1"{{.*}} "-triple" "spirv64-unknown-unknown"
// CHK-DEVICE-TRIPLE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHK-DEVICE-TRIPLE-SAME: "-fsycl-is-device"
// CHK-DEVICE-TRIPLE-SAME: "-O2"
// CHK-DEVICE-TRIPLE: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spirv64-unknown-unknown,arch=generic,kind=sycl"

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

/// Check for option incompatibility with -fsycl
// RUN: not %clang -### -fsycl -ffreestanding %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-ffreestanding
// RUN: not %clang -### -fsycl --offload-new-driver -static-libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INCOMPATIBILITY %s -DINCOMPATOPT=-static-libstdc++
// CHK-INCOMPATIBILITY: error: invalid argument '[[INCOMPATOPT]]' not allowed with '-fsycl'
