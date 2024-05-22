// RUN: %clang --target=ve-unknown-linux-gnu -### %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang --target=ve-unknown-linux-gnu -### %s -mvevpu -mno-vevpu 2>&1 | FileCheck %s -check-prefix=NO-VEVPU

// DEFAULT: "-target-feature" "+vpu"
// NO-VEVPU: "-target-feature" "-vpu"
