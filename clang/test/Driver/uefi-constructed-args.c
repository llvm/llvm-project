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

// RUN: %clang -### --target=loongarch64-unknown-uefi -g -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=LA64 %s
// RUN: %clang_cl -### --target=loongarch64-unknown-uefi -g -- %s 2>&1 \
// RUN:     | FileCheck -check-prefixes=LA64 %s
// LA64: "-cc1"
// LA64-SAME: "-triple" "loongarch64-unknown-uefi"
// LA64-SAME: "-mrelocation-model" "pic" "-pic-level" "2"
// LA64-SAME: "-mframe-pointer=all"
// LA64-SAME: "-fms-extensions"
// LA64-NEXT: "/nologo"
// LA64-SAME: "/subsystem:efi_application"
// LA64-SAME: "/entry:EfiMain"
// LA64-SAME: "/tsaware:no"
// LA64-SAME: "/debug"
