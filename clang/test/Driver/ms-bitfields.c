// RUN: %clang -### --target=x86_64-linux-gnu %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-LAYOUT
// RUN: %clang -### --target=x86_64-windows-gnu %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-LAYOUT
// RUN: %clang -### --target=x86_64-windows-msvc %s 2>&1 | FileCheck %s -check-prefix=DEFAULT-LAYOUT
// RUN: %clang -### -mms-bitfields %s 2>&1 | FileCheck %s -check-prefix=MICROSOFT-LAYOUT
// RUN: %clang -### -mno-ms-bitfields %s 2>&1 | FileCheck %s -check-prefix=ITANIUM-LAYOUT
// RUN: %clang -### -mno-ms-bitfields -mms-bitfields %s 2>&1 | FileCheck %s -check-prefix=MICROSOFT-LAYOUT
// RUN: %clang -### -mms-bitfields -mno-ms-bitfields %s 2>&1 | FileCheck %s -check-prefix=ITANIUM-LAYOUT

// DEFAULT-LAYOUT-NOT: -fms-layout-compatibility=itanium
// DEFAULT-LAYOUT-NOT: -fms-layout-compatibility=microsoft
// MICROSOFT-LAYOUT: -fms-layout-compatibility=microsoft
// MICROSOFT-LAYOUT-NOT: -fms-layout-compatibility=itanium
// ITANIUM-LAYOUT: -fms-layout-compatibility=itanium
// ITANIUM-LAYOUT-NOT: -fms-layout-compatibility=microsoft

