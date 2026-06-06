// RUN: %clang -std=c++20 -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -std=c++20 -fno-skip-odr-check-in-gmf -### -c %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=UNUSED
// RUN: %clang -std=c++20 -Xclang -fno-skip-odr-check-in-gmf -### -c %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NO-SKIP

// CHECK: -fskip-odr-check-in-gmf
// UNUSED: warning: argument unused during compilation: '-fno-skip-odr-check-in-gmf'
// UNUSED-NOT: -fno-skip-odr-check-in-gmf
// NO-SKIP: -fskip-odr-check-in-gmf{{.*}}-fno-skip-odr-check-in-gmf
