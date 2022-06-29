// Test running -fdepscan.
//
// REQUIRES: system-darwin, clang-cc1daemon

// RUN: %clang -cc1depscand -start %{clang-daemon-dir}/fdepscan-daemon
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=daemon -fdepscan-daemon=%{clang-daemon-dir}/fdepscan-daemon -fsyntax-only -x c %s
// RUN: %clang -cc1depscand -shutdown %{clang-daemon-dir}/fdepscan-daemon

#include "test.h"

int func(void);
