// Test passing of -Xarch_<arch> <option> to SYCL offload compilations.

// Verify that -Xarch_spirv64 forwards options to the SYCL device compilation
// and clang-linker-wrapper call.
// RUN: %clang -fsycl -Xarch_spirv64 -O3 -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SYCL-DEVICE-O3 %s
// SYCL-DEVICE-O3: "-triple" "spirv64-unknown-unknown" "-O3"{{.*}} "-fsycl-is-device"
// SYCL-DEVICE-O3: {{"[^"]*clang-linker-wrapper[^"]*".* "--device-compiler=spirv64-unknown-unknown=-O3"}}

// Verify that `-Xarch_spirv64` forwards libraries to the device linker.
// RUN: %clang -fsycl -Xarch_spirv64 -Wl,-lfoo -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=DEVICE-LINKER %s
// DEVICE-LINKER: {{"[^"]*clang-linker-wrapper[^"]*".* "--device-linker=spirv64-unknown-unknown=-lfoo"}}
