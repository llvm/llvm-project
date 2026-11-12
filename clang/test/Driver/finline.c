/// -fno-inline overrides -finline-functions/-finline-hint-functions.
// RUN: %clang -### -c --target=x86_64-apple-darwin10 -O2 -fno-inline -fno-inline-functions %s 2>&1 | FileCheck %s --check-prefix=NOINLINE
// RUN: %clang -### -c --target=x86_64 -O2 -finline -fno-inline -finline-functions %s 2>&1 | FileCheck %s --check-prefix=NOINLINE
// NOINLINE-NOT: "-finline-functions"
// NOINLINE:     "-fno-inline"
// NOINLINE-NOT: "-finline-functions"

/// -finline overrides -finline-functions.
// RUN: %clang -### -c --target=x86_64 -O2 -fno-inline -finline -finline-functions %s 2>&1 | FileCheck %s --check-prefix=INLINE
// INLINE-NOT: "-finline-functions"
// INLINE-NOT: "-fno-inline"
// INLINE-NOT: "-finline"

// RUN: %clang -### -c --target=aarch64 -O2 -finline-functions %s 2>&1 | FileCheck %s --check-prefix=INLINE-FUNCTIONS
// INLINE-FUNCTIONS: "-finline-functions"
