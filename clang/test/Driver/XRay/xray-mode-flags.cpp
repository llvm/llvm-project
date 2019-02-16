// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=xray-fdr %s -### \
// RUN:     2>&1 | FileCheck --check-prefix FDR %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=xray-basic %s \
// RUN:     -### 2>&1 | FileCheck --check-prefix BASIC %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=all -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes FDR,BASIC %s
// RUN: %clang -v -o /dev/null -fxray-instrument \
// RUN:     -fxray-modes=xray-fdr,xray-basic -### %s 2>&1 | \
// RUN:     FileCheck --check-prefixes FDR,BASIC %s
// RUN: %clang -v -o /dev/null -fxray-instrument \
// RUN:     -fxray-modes=xray-fdr -fxray-modes=xray-basic -### %s 2>&1 | \
// RUN:     FileCheck --check-prefixes FDR,BASIC %s
// RUN: %clang -v -o /dev/null -fxray-instrument -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes FDR,BASIC %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=none -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes NONE %s
//
// We also should support overriding the modes in an additive manner.
//
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=none,xray-fdr \
// RUN:     -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes FDR %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=xray-fdr,none \
// RUN:     -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes NONE %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=none,all \
// RUN:     -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes FDR,BASIC %s
//
// We also should support having the individual modes be concatenated.
//
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=none \
// RUN:     -fxray-modes=xray-fdr \
// RUN:     -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes FDR %s
//
// Order also matters.
//
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-modes=xray-fdr \
// RUN:     -fxray-modes=none \
// RUN:     -### %s \
// RUN:     2>&1 | FileCheck --check-prefixes NONE %s

// BASIC: libclang_rt.xray-basic
// FDR: libclang_rt.xray-fdr
// NONE-NOT: libclang_rt.xray-basic
// NONE-NOT: libclang_rt.xray-fdr
// REQUIRES-ANY: linux, freebsd
// REQUIRES-ANY: amd64, x86_64, x86_64h, arm, aarch64, arm64
