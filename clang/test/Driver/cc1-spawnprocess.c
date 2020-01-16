// RUN: %clang -fintegrated-cc1 -### %s 2>&1 | FileCheck %s --check-prefix=YES
// RUN: %clang -fno-integrated-cc1 -### %s 2>&1 | FileCheck %s --check-prefix=NO

// RUN: %clang -fintegrated-cc1 -fno-integrated-cc1 -### %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NO
// RUN: %clang -fno-integrated-cc1 -fintegrated-cc1 -### %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=YES

// RUN: %clang_cl -fintegrated-cc1 -### -- %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=YES
// RUN: %clang_cl -fno-integrated-cc1 -### -- %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NO

// RUN: env CCC_OVERRIDE_OPTIONS=+-fintegrated-cc1 \
// RUN:     %clang -fintegrated-cc1 -### %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=YES
// RUN: env CCC_OVERRIDE_OPTIONS=+-fno-integrated-cc1 \
// RUN:     %clang -fintegrated-cc1 -### %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=NO

// YES: (in-process)
// NO-NOT: (in-process)
