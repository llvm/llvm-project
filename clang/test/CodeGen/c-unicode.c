// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64--linux-gnu -S %s -o - | FileCheck %s -check-prefix=ALLOWED
// RUN: not %clang --target=x86_64--linux-gnu -std=c89 -S %s -o - 2>&1 | FileCheck %s -check-prefix=DENIED
int \uaccess = 0;
// ALLOWED: "곎ss":
// ALLOWED-NOT: "\uaccess":
// DENIED: warning: universal character names are only valid in C99 or C++; treating as '\' followed by identifier [-Wunicode]
// DENIED: error: expected identifier or '('
