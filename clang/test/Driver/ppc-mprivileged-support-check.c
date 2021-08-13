// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -mcpu=pwr10 -mprivileged %s -o - | FileCheck %s --check-prefix=HASPRIV
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -mcpu=power10 -mprivileged %s -o - | FileCheck %s --check-prefix=HASPRIV
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -mcpu=pwr9 -mprivileged %s -o - | FileCheck %s --check-prefix=HASPRIV
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -mcpu=power9 -mprivileged %s -o - | FileCheck %s --check-prefix=HASPRIV
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -mcpu=pwr8 -mprivileged %s -o - | FileCheck %s --check-prefix=HASPRIV
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm \
// RUN:   -mcpu=power8 -mprivileged %s -o - | FileCheck %s --check-prefix=HASPRIV

// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr7 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=NOPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power7 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=NOPRIV

// __SP__ TODO: Fix this test case to check for the attribute.
int test() {
  return 0;
}

// HASPRIV: test() #0 {
// HASPRIV: attributes #0 = {
// HASPRIV-SAME: +privileged

// NOPRIV: option '-mprivileged' cannot be specified with

