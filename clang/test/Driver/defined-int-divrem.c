// The driver forwards exactly the spelling that was given to cc1; the default
// (on for OpenCL) is computed by cc1 itself.

// RUN: %clang -### -c -target x86_64-unknown-unknown -fdefined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=ON
// RUN: %clang -### -c -target x86_64-unknown-unknown -fno-defined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c -target x86_64-unknown-unknown %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64-unknown-unknown -x cl -fno-defined-int-divrem %s 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clang -### -c -target x86_64-unknown-unknown -x cl %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

// ON: "-fdefined-int-divrem"
// ON-NOT: "-fno-defined-int-divrem"

// OFF: "-fno-defined-int-divrem"
// OFF-NOT: "-fdefined-int-divrem"

// Nothing is forwarded by default, in C or in OpenCL.
// DEFAULT-NOT: "-f{{(no-)?}}defined-int-divrem"
