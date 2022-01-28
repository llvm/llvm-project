// RUN: %clang -### -S %s        2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s
// RUN: %clang -### -S %s -g -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G %s

// Assert that the toolchains which should default to a lower Dwarf version do so.
// RUN: %clang -### -S %s -g -target x86_64-apple-darwin8 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF2 %s
// RUN: %clang -### -S %s -g -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF2 %s
// RUN: %clang -### -S %s -g -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF2 %s

// 'g0' is the default. Just basic correctness check that it does nothing
// RUN: %clang -### -S %s -g0    2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s

// And check that the last of -g or -g0 wins.
// RUN: %clang -### -S %s -g -g0 2>&1 | FileCheck --check-prefix=CHECK-WITHOUT-G %s

// These should be semantically the same as not having given 'g0' at all,
// as the last 'g' option wins.
//
// RUN: %clang -### -S %s -g0 -g -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G %s
// RUN: %clang -### -S %s -g0 -g -target x86_64-apple-darwin8 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-STANDALONE %s
// RUN: %clang -### -S %s -g0 -g -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF2 %s
// RUN: %clang -### -S %s -g0 -g -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF2 %s
// RUN: %clang -### -S %s -g0 -g -target i386-pc-solaris 2>&1 \
// RUN:             | FileCheck --check-prefix=CHECK-WITH-G-DWARF2 %s

// CHECK-WITHOUT-G-NOT: -debug-info-kind
// CHECK-WITH-G: "-debug-info-kind=constructor"
// CHECK-WITH-G: "-dwarf-version=5"
// CHECK-WITH-G-DWARF2: "-dwarf-version=2"

// CHECK-WITH-G-STANDALONE: "-debug-info-kind=standalone"
// CHECK-WITH-G-STANDALONE: "-dwarf-version=2"
