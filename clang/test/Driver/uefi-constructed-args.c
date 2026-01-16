// RUN: %clang -### --target=x86_64-unknown-uefi -g -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK %s
// RUN: %clang_cl -### --target=x86_64-unknown-uefi -g -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK %s
// CHECK: "-cc1"
// CHECK-SAME: "-triple" "x86_64-unknown-uefi"
// CHECK-SAME: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-SAME: "-mframe-pointer=all"
// CHECK-SAME: "-fms-extensions"
// CHECK-NEXT: "/nologo"
// CHECK-SAME: "/subsystem:efi_application"
// CHECK-SAME: "/entry:EfiMain"
// CHECK-SAME: "/tsaware:no"
// CHECK-SAME: "/debug"

// RUN: %clang -### --target=x86_64-unknown-uefi -print-search-dirs 2>&1 \
// RUN:     | FileCheck -check-prefixes=PROGPATH %s
// PROGPATH: InstalledDir: [[DRIVER_INSTALLED_DIR:.*]]
// PROGPATH: programs: =[[DRIVER_INSTALLED_DIR]]
