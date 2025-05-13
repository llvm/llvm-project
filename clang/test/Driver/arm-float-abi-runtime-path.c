/// Check that libraries built with the per target runtime directory layout
/// are selected correctly when using variations of Arm triples.

// RUN: %clang %s --target=arm-unknown-linux-gnueabihf -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=ARMHF %s
/// "armv7l" should be normalised to just "arm".
// RUN: %clang %s --target=armv7l-unknown-linux-gnueabihf -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=ARMHF %s

// RUN: %clang %s --target=arm-unknown-linux-gnueabi -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=ARM %s
// RUN: %clang %s --target=armv7l-unknown-linux-gnueabi -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=ARM %s

/// armeb triples should be unmodified.
// RUN: %clang %s --target=armeb-unknown-linux-gnueabihf -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=ARMEBHF %s
// RUN: %clang %s --target=armeb-unknown-linux-gnueabi -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=ARMEB %s

// RUN: %clang %s --target=arm-pc-windows-msvc -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=WINDOWS %s
/// armhf-pc... isn't recognised so just check that the float-abi option is ignored
// RUN: %clang %s --target=arm-pc-windows-msvc -mfloat-abi=hard -print-runtime-dir \
// RUN:        -resource-dir=%S/Inputs/arm_float_abi_runtime_path 2>&1 | FileCheck -check-prefix=WINDOWS %s

// ARMHF:   lib{{/|\\}}arm-unknown-linux-gnueabihf{{$}}
// ARM:     lib{{/|\\}}arm-unknown-linux-gnueabi{{$}}
// ARMEBHF: lib{{/|\\}}armeb-unknown-linux-gnueabihf{{$}}
// ARMEB:   lib{{/|\\}}armeb-unknown-linux-gnueabi{{$}}
// WINDOWS: lib{{/|\\}}arm-pc-windows-msvc{{$}}
