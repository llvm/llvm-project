// The driver forwards the last of -f[no-]defined-int-divrem to cc1; cc1
// computes the default (on for OpenCL) itself.

// RUN: %clang -### -c --target=x86_64-unknown-linux -fdefined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### -c --target=x86_64-unknown-linux -fno-defined-int-divrem -fdefined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### -c --target=x86_64-unknown-linux -fno-defined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c --target=x86_64-unknown-linux -fdefined-int-divrem -fno-defined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c --target=x86_64-unknown-linux -x cl -fno-defined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c --target=x86_64-unknown-linux %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c --target=x86_64-unknown-linux -x cl %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c --target=spirv64 -x cl %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

// ON:      "-fdefined-int-divrem"
// ON-NOT:  "-fno-defined-int-divrem"
// OFF:     "-fno-defined-int-divrem"
// OFF-NOT: "-fdefined-int-divrem"
// DEFAULT-NOT: defined-int-divrem"
