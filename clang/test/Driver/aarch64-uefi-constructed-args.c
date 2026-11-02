// RUN: %clang -### --target=aarch64-unknown-uefi -mno-incremental-linker-compatible -g -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK %s
// RUN: %clang_cl -### --target=aarch64-unknown-uefi -mno-incremental-linker-compatible -g -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK %s
// CHECK: "-cc1"
// CHECK-SAME: "-triple" "aarch64-unknown-uefi"
// CHECK-SAME: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-SAME: "-mframe-pointer=non-leaf"
// CHECK-SAME: "-fms-extensions"
// CHECK-NEXT: "/nologo"
// CHECK-SAME: "/subsystem:efi_application"
// CHECK-SAME: "/entry:EfiMain"
// CHECK-SAME: "/tsaware:no"
// CHECK-SAME: "/debug"
// CHECK-SAME: "/Brepro"

// As on Windows ARM64, the frame chain must stay valid even when frame
// pointers are omitted, and char is signed.
// RUN: %clang -### --target=aarch64-unknown-uefi -fomit-frame-pointer -c -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=DEFAULTS %s
// DEFAULTS: "-cc1"
// DEFAULTS-SAME: "-mframe-pointer=reserved"
// DEFAULTS-NOT: "-fno-signed-char"

// RUN: %clang -### --target=aarch64-unknown-uefi -print-search-dirs 2>&1 \
// RUN:     | FileCheck -check-prefixes=PROGPATH %s
// PROGPATH: InstalledDir: [[DRIVER_INSTALLED_DIR:.*]]
// PROGPATH: programs: =[[DRIVER_INSTALLED_DIR]]
