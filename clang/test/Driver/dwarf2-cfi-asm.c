// RUN: %clang -target x86_64-apple-darwin -### -S -integrated-as %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-DARWIN-MC-DEFAULT %s
// RUN: %clang -target x86_64-apple-darwin -### -S -integrated-as -fdwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-DARWIN-MC-CFI %s
// RUN: %clang -target x86_64-apple-darwin -### -S -integrated-as -fno-dwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-DARWIN-MC-NOCFI %s

// RUN: %clang -target x86_64-apple-darwin -### -S -no-integrated-as %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-DARWIN-AS-DEFAULT %s
// RUN: %clang -target x86_64-apple-darwin -### -S -no-integrated-as -fdwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-DARWIN-AS-CFI %s
// RUN: %clang -target x86_64-apple-darwin -### -S -no-integrated-as -fno-dwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-DARWIN-AS-NOCFI %s


// RUN: %clang -target x86_64-pc-linux -### -S -integrated-as %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-LINUX-MC-DEFAULT %s
// RUN: %clang -target x86_64-pc-linux -### -S -integrated-as -fdwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-LINUX-MC-CFI %s
// RUN: %clang -target x86_64-pc-linux -### -S -integrated-as -fno-dwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-LINUX-MC-NOCFI %s

// RUN: %clang -target x86_64-pc-linux -### -S -no-integrated-as %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-LINUX-AS-DEFAULT %s
// RUN: %clang -target x86_64-pc-linux -### -S -no-integrated-as -fdwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-LINUX-AS-CFI %s
// RUN: %clang -target x86_64-pc-linux -### -S -no-integrated-as -fno-dwarf2-cfi-asm %s -o %t.s 2>&1 | FileCheck -check-prefix=CHECK-LINUX-AS-NOCFI %s



// CHECK-DARWIN-MC-DEFAULT-NOT: -fno-dwarf2-cfi-asm
// CHECK-DARWIN-MC-CFI-NOT: -fno-dwarf2-cfi-asm
// CHECK-DARWIN-MC-NOCFI: -fno-dwarf2-cfi-asm

// CHECK-DARWIN-AS-DEFAULT: -fno-dwarf2-cfi-asm
// CHECK-DARWIN-AS-CFI-NOT: -fno-dwarf2-cfi-asm
// CHECK-DARWIN-AS-NOCFI: -fno-dwarf2-cfi-asm


// CHECK-LINUX-MC-DEFAULT-NOT: -fno-dwarf2-cfi-asmx
// CHECK-LINUX-MC-CFI-NOT: -fno-dwarf2-cfi-asm
// CHECK-LINUX-MC-NOCFI: -fno-dwarf2-cfi-asm

// CHECK-LINUX-AS-DEFAULT-NOT: -fno-dwarf2-cfi-asm
// CHECK-LINUX-AS-CFI-NOT: -fno-dwarf2-cfi-asm
// CHECK-LINUX-AS-NOCFI: -fno-dwarf2-cfi-asm
