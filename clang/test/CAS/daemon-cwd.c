// Test running -fdepscan with a daemon launched from different directory.
//
// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t && mkdir -p %t/include
// RUN: cp %S/Inputs/test.h %t/include
// RUN: echo "#!/bin/sh" >> %t/cmd.sh
// RUN: echo cd %t >> %t/cmd.sh
// RUN: echo %clang -target x86_64-apple-macos11 -fdepscan=daemon         \
// RUN:    -fdepscan-prefix-map=%S=/^source                               \
// RUN:    -fdepscan-prefix-map=%t=/^build                                \
// RUN:    -fdepscan-prefix-map-toolchain=/^toolchain                     \
// RUN:    -fdepscan-daemon=%{clang-daemon-dir}/%basename_t               \
// RUN:    -Xclang -fcas-path -Xclang %t/cas                              \
// RUN:    -MD -MF %t/test.d -Iinclude                                    \
// RUN:    -fsyntax-only -x c %s >> %t/cmd.sh
// RUN: chmod +x %t/cmd.sh

// RUN: %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t      \
// RUN:   -cas-args -fcas-path %t/cas -- %t/cmd.sh
// RUN: (cd %t && %clang -target x86_64-apple-macos11 -MD -MF %t/test2.d  \
// RUN:    -Iinclude -fsyntax-only -x c %s)
// RUN: diff %t/test.d %t/test2.d

#include "test.h"

int func(void);
