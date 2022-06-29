// Test running -fdepscan with a daemon launched from different directory.
//
// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t && mkdir -p %t/include
// RUN: cp %S/Inputs/test.h %t/include
// RUN: %clang -cc1depscand -start %{clang-daemon-dir}/daemon-cwd
// RUN: (cd %t && %clang -target x86_64-apple-macos11 -fdepscan=daemon    \
// RUN:    -fdepscan-prefix-map=%S=/^source                               \
// RUN:    -fdepscan-prefix-map=%t=/^build                                \
// RUN:    -fdepscan-prefix-map-toolchain=/^toolchain                     \
// RUN:    -fdepscan-daemon=%{clang-daemon-dir}/daemon-cwd                \
// RUN:    -MD -MF %t/test.d -Iinclude                                    \
// RUN:    -fsyntax-only -x c %s)
// RUN: %clang -cc1depscand -shutdown %{clang-daemon-dir}/daemon-cwd
// RUN: %clang -target x86_64-apple-macos11 -MD -MF %t/test2.d            \
// RUN:    -I %t/include -fsyntax-only -x c %s
// RUN: diff %t/test.d %t/test2.d

#include "test.h"

int func(void);
