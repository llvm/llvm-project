// RUN: %clang --target=sparc64 -mcmodel=medlow %s -### 2>&1 | FileCheck -check-prefix=MEDLOW %s
// RUN: %clang --target=sparc64 -mcmodel=medmid %s -### 2>&1 | FileCheck -check-prefix=MEDMID %s
// RUN: %clang --target=sparc64 -mcmodel=medany %s -### 2>&1 | FileCheck -check-prefix=MEDANY %s
// MEDLOW: "-mcmodel=small"
// MEDMID: "-mcmodel=medium"
// MEDANY: "-mcmodel=large"
