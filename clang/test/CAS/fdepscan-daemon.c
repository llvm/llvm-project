// Test running -fdepscan.
//
// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t
// RUN: %clang -cc1depscand -start %{clang-daemon-dir}/fdepscan-daemon -cas-args -fcas-path %t/cas -faction-cache-path %t/cache
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs \
// RUN:   -fdepscan=daemon -fdepscan-daemon=%{clang-daemon-dir}/fdepscan-daemon -fsyntax-only -x c %s
// RUN: %clang -cc1depscand -shutdown %{clang-daemon-dir}/fdepscan-daemon

#include "test.h"

int func(void);
