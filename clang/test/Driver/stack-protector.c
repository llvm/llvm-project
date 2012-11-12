// RUN: %clang -fno-stack-protector -### %s 2>&1 | FileCheck %s -check-prefix=NOSSP
// NOSSP-NOT: "-stack-protector" "1"
// NOSSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector -### %s 2>&1 | FileCheck %s -check-prefix=SSP
// SSP: "-stack-protector" "1"
// SSP-NOT: "-stack-protector-buffer-size" 

// RUN: %clang -fstack-protector --param ssp-buffer-size=16 -### %s 2>&1 | FileCheck %s -check-prefix=SSP-BUF
// SSP-BUF: "-stack-protector" "1"
// SSP-BUF: "-stack-protector-buffer-size" "16" 
