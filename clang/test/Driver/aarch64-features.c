// RUN: %clang -target aarch64-none-linux-gnu -### %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target arm64-none-linux-gnu -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: "-munwind-tables"

// The AArch64 PCS states that chars should be unsigned.
// CHECK: fno-signed-char

