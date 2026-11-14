// Mach-O objects with DWARF 5 or later relocate offsets between DWARF sections,
// except for the Apple operating systems below. This applies to code
// generation, the integrated assembler, and LTO code generation in the linker.

// RUN: %clang -### --target=arm64-apple-none-macho -g -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1
// RUN: %clang -### --target=arm64-apple-firmware -g -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1
// RUN: %clang -### --target=arm64-apple-bridgeos9.0 -g -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1
// RUN: %clang -### --target=arm64-apple-none-macho -g -c -x assembler %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1AS
// RUN: %clang -### --target=arm64-apple-none-macho -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK

// RUN: %clang -### --target=arm64-apple-none-macho -g -gdwarf-4 -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-none-macho -gdwarf-4 -c -x assembler %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-macos15 -g -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-ios17 -g -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-tvos17 -g -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-watchos10 -g -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-xros1 -g -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=arm64-apple-driverkit23 -g -flto %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: %clang -### --target=x86_64-linux-gnu -g -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE

// CC1:   "-cc1" {{.*}}"-grelocatable-dwarf"
// CC1AS: "-cc1as" {{.*}}"-grelocatable-dwarf"
// LINK:  "{{.*}}ld{{(.exe)?}}" {{.*}}"-mllvm" "-macho-relocatable-dwarf"
// NONE-NOT: "-grelocatable-dwarf"
// NONE-NOT: "-macho-relocatable-dwarf"
