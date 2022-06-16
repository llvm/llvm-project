// Test running -fdepscan.
//
// REQUIRES: system-darwin, short-build-dir-path

// RUN: %clang -cc1depscand -start %t/depscand
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=daemon -fdepscan-daemon=%t/depscand -fsyntax-only -x c %s
// RUN: %clang -cc1depscand -shutdown %t/depscand

#include "test.h"

int func(void);
