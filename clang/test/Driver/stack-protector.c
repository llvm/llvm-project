// RUN: %clang -fno-stack-protector -### %s 2>&1 | FileCheck %s -check-prefix=NOSSP
// NOSSP-NOT: "-stack-protector"
// NOSSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -target i386-unknown-linux -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP
// SSP: "-stack-protector" "1"
// SSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -target i386-unknown-linux -fstack-protector --param ssp-buffer-size=16 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-BUF
// SSP-BUF: "-stack-protector" "1"
// SSP-BUF: "-stack-protector-buffer-size" "16" 

// RUN: %clang -target i386-pc-openbsd -### %s 2>&1 | FileCheck %s -check-prefix=OPENBSD
// OPENBSD: "-stack-protector" "2"

// RUN: %clang -target i386-pc-openbsd -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=OPENBSD_SPS
// OPENBSD_SPS: "-stack-protector" "2"

// RUN: %clang -fstack-protector-strong -### %s 2>&1 | FileCheck %s -check-prefix=SSP-STRONG
// SSP-STRONG: "-stack-protector" "2"
// SSP-STRONG-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector-all -### %s 2>&1 | FileCheck %s -check-prefix=SSP-ALL
// SSP-ALL: "-stack-protector" "3"
// SSP-ALL-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -target x86_64-scei-ps4 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-PS4
// RUN: %clang -target x86_64-scei-ps4 -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP-PS4
// SSP-PS4: "-stack-protector" "2"
// SSP-PS4-NOT: "-stack-protector-buffer-size"

// RUN: %clang -target x86_64-scei-ps4 -fstack-protector --param ssp-buffer-size=16 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-PS4-BUF
// SSP-PS4-BUF: "-stack-protector" "2"
// SSP-PS4-BUF: "-stack-protector-buffer-size" "16"
