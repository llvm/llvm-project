// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=xray-fdr %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FDR %s
// RUN: %clang -### --target=aarch64-linux-gnu -fxray-instrument -fxray-modes=xray-basic %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s
// RUN: %clang -### --target=aarch64-linux-gnu -fxray-instrument %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FDR,BASIC %s
// RUN: %clang -### --target=s390x-linux-gnu -fxray-instrument -fxray-modes=xray-basic %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s
// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=all %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FDR,BASIC %s
// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=xray-fdr,xray-basic %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FDR,BASIC %s
// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=xray-fdr -fxray-modes=xray-basic %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FDR,BASIC %s
// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=none %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NONE %s
//
// We also should support overriding the modes in an additive manner.

// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=none,xray-fdr %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FDR %s
// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=all,none %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NONE %s

// We also should support having the individual modes be concatenated.

// RUN: %clang -### --target=x86_64-linux-gnu -fxray-instrument -fxray-modes=none -fxray-modes=xray-fdr %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FDR %s

// BASIC: libclang_rt.xray-basic
// FDR: libclang_rt.xray-fdr
// NONE-NOT: libclang_rt.xray-basic
// NONE-NOT: libclang_rt.xray-fdr
