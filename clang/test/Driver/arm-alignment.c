// RUN: %clang -target arm-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target armv6-apple-darwin -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target armv6-netbsd-eabi -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target armv7-unknown-linux -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target armv7-unknown-nacl-gnueabihf -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-ARM < %t %s

// RUN: %clang -target aarch64-none-gnueabi -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mstrict-align -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mno-unaligned-access -munaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNALIGNED-AARCH64 < %t %s

// CHECK-UNALIGNED-ARM-NOT: "-target-feature" "+strict-align"
// CHECK-UNALIGNED-AARCH64-NOT: "-target-feature" "+strict-align"


// RUN: %clang -target arm-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target arm-none-gnueabi -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target armv5-apple-darwin -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target armv5t-netbsd-eabi -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target armv6-unknown-linux -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target armv6-unknown-nacl-gnueabihf -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target armv6m-apple-darwin -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target armv6m-netbsd-eabi -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-ARM < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -munaligned-access -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -munaligned-access -mstrict-align -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-none-gnueabi -mkernel -mno-unaligned-access -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// RUN: %clang -target aarch64-unknown-openbsd -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ALIGNED-AARCH64 < %t %s

// CHECK-ALIGNED-ARM: "-target-feature" "+strict-align"
// CHECK-ALIGNED-AARCH64: "-target-feature" "+strict-align"

// Make sure that v6M cores and v8M Baseline always trigger the unsupported
// aligned accesses error for all supported architecture triples.
// RUN: not %clang -c -target thumbv6m-none-gnueabi -mcpu=cortex-m0 -munaligned-access %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-UNALIGN-NOT-SUPPORTED %s
// RUN: not %clang -c -target thumb-none-gnueabi -mcpu=cortex-m0 -munaligned-access %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-UNALIGN-NOT-SUPPORTED %s
// RUN: not %clang -c -target thumbv8m.base-none-gnueabi -munaligned-access %s 2>&1 | \
// RUN:   FileCheck --check-prefix CHECK-UNALIGN-NOT-SUPPORTED %s

// CHECK-UNALIGN-NOT-SUPPORTED: error: the {{.*}} sub-architecture does not support unaligned accesses
