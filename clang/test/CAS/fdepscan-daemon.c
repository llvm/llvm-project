// Test running -fdepscan.
//
// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t
// RUN: %clang -cc1depscand -execute %{clang-daemon-dir}/%basename_t -cas-args -fcas-path %t/cas -- \
// RUN:   %clang -target x86_64-apple-macos11 -I %S/Inputs \
// RUN:     -Xclang -fcas-path -Xclang %t/cas \
// RUN:     -fdepscan=daemon -fdepscan-daemon=%{clang-daemon-dir}/%basename_t -fsyntax-only -x c %s

#include "test.h"

int func(void);
