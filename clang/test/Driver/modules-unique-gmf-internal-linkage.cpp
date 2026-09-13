// RUN: %clang -std=c++20 -fmodules-unique-gmf-internal-linkage -### -c %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=ENABLE
// RUN: %clang -std=c++20 -fno-modules-unique-gmf-internal-linkage -### -c %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=DISABLE
// RUN: %clang -std=c++20 -fno-modules-unique-gmf-internal-linkage \
// RUN:   -fmodules-unique-gmf-internal-linkage -### -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LAST-ENABLE
// RUN: %clang -std=c++20 -fmodules-unique-gmf-internal-linkage \
// RUN:   -fno-modules-unique-gmf-internal-linkage -### -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LAST-DISABLE

// ENABLE: "-cc1"
// ENABLE-SAME: "-fmodules-unique-gmf-internal-linkage"
// ENABLE-NOT: argument unused during compilation

// DISABLE: "-cc1"
// DISABLE-SAME: "-fno-modules-unique-gmf-internal-linkage"
// DISABLE-NOT: argument unused during compilation

// LAST-ENABLE: "-cc1"
// LAST-ENABLE-SAME: "-fmodules-unique-gmf-internal-linkage"
// LAST-ENABLE-NOT: "-fno-modules-unique-gmf-internal-linkage"

// LAST-DISABLE: "-cc1"
// LAST-DISABLE-SAME: "-fno-modules-unique-gmf-internal-linkage"
// LAST-DISABLE-NOT: "-fmodules-unique-gmf-internal-linkage"

int main() { return 0; }
