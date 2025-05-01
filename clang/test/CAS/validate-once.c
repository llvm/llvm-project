// RUN: rm -rf %t

// RUN: llvm-cas --cas %t/cas --ingest %s
// RUN: mv %t/cas/v1.1/v8.data %t/cas/v1.1/v8.data.bak

// RUN: %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t -cas-args -fcas-path %t/cas -- \
// RUN:   %clang -target x86_64-apple-macos11 -I %S/Inputs \
// RUN:     -Xclang -fcas-path -Xclang %t/cas \
// RUN:     -fdepscan=daemon -fdepscan-daemon=%{clang-daemon-dir}/%basename_t -fsyntax-only -x c %s

// RUN: ls %t/cas/corrupt.0.v1.1

// RUN: llvm-cas --cas %t/cas --validate-if-needed | FileCheck %s -check-prefix=SKIPPED
// SKIPPED: validation skipped

#include "test.h"

int func(void);
