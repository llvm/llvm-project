// With LLD, -demangle is always passed to the linker on Darwin.

// RUN: %clang --target=x86_64-apple-darwin -### -fuse-ld=lld \
// RUN:   -B%S/Inputs/lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=x86_64-apple-darwin -### -fuse-ld=lld \
// RUN:   --ld-path=%S/Inputs/lld/ld64.lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=x86_64-apple-darwin -### \
// RUN:   --ld-path=%S/Inputs/lld/ld64.lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=x86_64-apple-darwin -### -Wno-fuse-ld-path \
// RUN:   -fuse-ld=%S/Inputs/lld/ld64.lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang --target=x86_64-apple-darwin -### \
// RUN:   --ld-path=%S/Inputs/lld/ld.lld -mlinker-version=0 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-DEMANGLE %s

// CHECK: "-demangle"
// NO-DEMANGLE-NOT: "-demangle"
