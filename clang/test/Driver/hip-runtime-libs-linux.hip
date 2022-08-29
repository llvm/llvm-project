// UNSUPPORTED: system-windows

// RUN:  touch %t.o

// Test HIP runtime lib args specified by --rocm-path.
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-PATH %s

// Test HIP runtime lib args specified by environment variable ROCM_PATH.
// RUN: env ROCM_PATH=%S/Inputs/rocm %clang -### --hip-link \
// RUN:   --target=x86_64-linux-gnu %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-PATH %s

// Test detecting latest /opt/rocm-{release} directory.
// RUN: rm -rf %T/opt
// RUN: mkdir -p %T/opt
// RUN: cp -r %S/Inputs/rocm %T/opt/rocm-3.9.0-1234
// RUN: cp -r %S/Inputs/rocm %T/opt/rocm-3.10.0
// RUN: %clang -### --hip-link --target=x86_64-linux-gnu \
// RUN:   --sysroot=%T %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-REL %s

// Test HIP runtime lib is not linked without --hip-link.
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOHIPRT %s

// Test HIP runtime lib is not linked with -nostdlib.
// RUN: %clang -### --hip-link -nostdlib --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOHIPRT %s

// Test HIP runtime lib is not linked with -no-hip-rt.
// RUN: %clang -### --hip-link -no-hip-rt --target=x86_64-linux-gnu \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NOHIPRT %s

// ROCM-PATH: "-L[[HIPRT:.*/Inputs/rocm/lib]]" "-rpath" "[[HIPRT]]" "-lamdhip64"
// ROCM-REL: "-L[[HIPRT:.*/opt/rocm-3.10.0/lib]]" "-rpath" "[[HIPRT]]" "-lamdhip64"
// NOHIPRT-NOT: "-L{{.*/Inputs/rocm/lib}}"
// NOHIPRT-NOT: "-rpath" "{{.*/Inputs/rocm/lib}}"
// NOHIPRT-NOT: "-lamdhip64"
