// Test running -fdepscan with a daemon launched from different directory.
//
// REQUIRES: clang-cc1daemon

// RUN: rm -rf %t && split-file %s %t

// RUN: %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t       \
// RUN:   -cas-args -fcas-path %t/cas --                                   \
// RUN:     %clang -target x86_64-apple-macos11 -fdepscan=daemon           \
// RUN:       -fdepscan-prefix-map=%S=/^source                             \
// RUN:       -fdepscan-prefix-map=%t=/^build                              \
// RUN:       -fdepscan-prefix-map-toolchain=/^toolchain                   \
// RUN:       -fdepscan-daemon=%{clang-daemon-dir}/%basename_t             \
// RUN:       -Xclang -fcas-path -Xclang %t/cas                            \
// RUN:       -working-directory %t -MD -MF %t/test.d -Iinclude            \
// RUN:       -fsyntax-only -x c %t/daemon-cwd.c
// RUN: %clang -target x86_64-apple-macos11 -working-directory %t          \
// RUN:   -MD -MF %t/test2.d -Iinclude -fsyntax-only -x c %t/daemon-cwd.c
// RUN: diff %t/test.d %t/test2.d

//--- daemon-cwd.c
#include "test.h"

int func(void);

//--- include/test.h
int test(void);
