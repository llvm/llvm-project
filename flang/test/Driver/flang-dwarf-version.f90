// RUN: %if !target={{.*aix.*}} %{ \
// RUN: %flang -### -S %s -g -gdwarf-5  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF5 %s \
// RUN: %}

// RUN: %if !target={{.*aix.*}} %{ \
// RUN: %flang -### -S %s -gdwarf-5  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF5 %s \
// RUN: %}

// RUN: %if !target={{.*aix.*}} %{ \
// RUN: %flang -### -S %s -g1 -gdwarf-5  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G1-DWARF5 %s \
// RUN: %}

// RUN: %flang -### -S %s -gdwarf-4  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF4 %s
// RUN: %flang -### -S %s -gdwarf-3  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF3 %s
// RUN: %flang -### -S %s -gdwarf-2  2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF2 %s

// Without an explicit -gdwarf-N, the toolchain default DWARF version is used.

// Linux.
// RUN: %flang -### -S %s -g --target=x86_64-unknown-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF5 %s
// RUN: %flang -### -S %s -g1 --target=x86_64-unknown-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G1-DWARF5 %s

// Android always uses DWARF 4.
// RUN: %flang -### -S %s -g --target=aarch64-unknown-linux-android21 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF4 %s

// Darwin derives the version from the OS version rather than using a constant.
// RUN: %flang -### -S %s -g --target=x86_64-apple-macosx15 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF5 %s
// RUN: %flang -### -S %s -g --target=x86_64-apple-macosx10.10 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF2 %s

// AIX.
// RUN: %flang -### -S %s -g --target=powerpc64-ibm-aix 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF3 %s

// OpenBSD.
// RUN: %flang -### -S %s -g --target=x86_64-unknown-openbsd 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF2 %s

// No debug info requested means no DWARF version is passed at all.
// RUN: %flang -### -S %s 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-NO-DWARF %s
// RUN: %flang -### -S %s -g0 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-NO-DWARF %s

// A version named explicitly is still passed on when debug info is switched
// off, as clang does.
// RUN: %flang -### -S %s -gdwarf-5 -g0 --target=x86_64-unknown-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-DWARF5-G0 %s

// CHECK-DWARF5: -debug-info-kind=standalone
// CHECK-DWARF5-SAME: -dwarf-version=5

// CHECK-WITH-G1-DWARF5: -debug-info-kind=line-tables-only
// CHECK-WITH-G1-DWARF5-SAME: -dwarf-version=5

// CHECK-DWARF4: -dwarf-version=4

// CHECK-DWARF3: -dwarf-version=3

// CHECK-DWARF2: -dwarf-version=2

// CHECK-NO-DWARF-NOT: -dwarf-version=

// CHECK-DWARF5-G0: -dwarf-version=5
