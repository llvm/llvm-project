// REQUIRES: lld
// Check that lld will emit dwarf aranges.

// RUN: %clang -### -target x86_64-unknown-linux -c -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=GARANGE %s
// RUN: %clang -### -target x86_64-unknown-linux -flto -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=LDGARANGE %s
// RUN: %clang -### -target x86_64-unknown-linux -flto=thin -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=LDGARANGE %s
// RUN: %clang -### -target x86_64-unknown-linux -fuse-ld=lld -flto -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=LLDGARANGE %s
// RUN: %clang -### -target x86_64-unknown-linux -fuse-ld=lld -flto=thin -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=LLDGARANGE %s
//
// GARANGE-DAG: -generate-arange-section
// LDGARANGE-NOT: {{".*lld.*"}} {{.*}} "-generate-arange-section"
// LLDGARANGE: {{".*lld.*"}} {{.*}} "-generate-arange-section"
