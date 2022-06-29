// Test running -fdepscan.
//
// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t-*.d %t.cas
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=daemon \
// RUN:   -E -MD -MF %t-daemon.d -x c %s -Xclang -fcas-path -Xclang %t.cas >/dev/null
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=inline \
// RUN:   -E -MD -MF %t-inline.d -x c %s -Xclang -fcas-path -Xclang %t.cas >/dev/null
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=auto \
// RUN:   -E -MD -MF %t-auto.d -x c %s -Xclang -fcas-path -Xclang %t.cas >/dev/null
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=off \
// RUN:   -E -MD -MF %t-off.d -x c %s -Xclang -fcas-path -Xclang %t.cas >/dev/null
//
// Check -fdepscan-share-related arguments are claimed.
// TODO: Check behaviour.
//
// RUN: %clang -target x86_64-apple-macos11 -I %S/Inputs -fdepscan=off \
// RUN:     -fdepscan-share-parent                                     \
// RUN:     -fdepscan-share-parent=                                    \
// RUN:     -fdepscan-share-parent=python                              \
// RUN:     -fdepscan-share=python                                     \
// RUN:     -fdepscan-share=                                           \
// RUN:     -fdepscan-share-stop=python                                \
// RUN:     -fdepscan-share-identifier                                 \
// RUN:     -fno-depscan-share                                         \
// RUN:     -fsyntax-only -x c %s                                      \
// RUN:     -Xclang -fcas-path -Xclang %t.cas                          \
// RUN: | FileCheck %s -allow-empty
// CHECK-NOT: warning:
//
// RUN: not %clang -target x86_64-apple-macos11 -I %S/Inputs \
// RUN:     -fdepscan-share-parents 2>&1                     \
// RUN: | FileCheck %s -check-prefix=BAD-SPELLING
// BAD-SPELLING: error: unknown argument '-fdepscan-share-parents'
//
// Check that the dependency files match.
// RUN: diff %t-off.d %t-daemon.d
// RUN: diff %t-off.d %t-inline.d
// RUN: diff %t-off.d %t-auto.d

#include "test.h"

int func(void);
