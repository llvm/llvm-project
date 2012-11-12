// REQUIRES: x86-registered-target

// RUN: %clang -target i386-apple-darwin -S -o - %s | \
// RUN:   FileCheck --check-prefix=DARWIN %s
// DARWIN: f0:
// DARWIN: pushl %ebp
// DARWIN: ret
// DARWIN: f1:
// DARWIN: pushl %ebp
// DARWIN: ret

// RUN: %clang -target i386-pc-linux-gnu -S -o - %s | \
// RUN:   FileCheck --check-prefix=LINUX %s
// LINUX: f0:
// LINUX-NOT: pushl %ebp
// LINUX: ret
// LINUX: f1:
// LINUX: pushl %ebp
// LINUX: ret

// RUN: %clang -target i386-darwin -S -o - -fomit-frame-pointer %s | \
// RUN:   FileCheck --check-prefix=OMIT_ALL %s
// OMIT_ALL: f0:
// OMIT_ALL-NOT: pushl %ebp
// OMIT_ALL: ret
// OMIT_ALL: f1:
// OMIT_ALL-NOT: pushl %ebp
// OMIT_ALL: ret

// RUN: %clang -target i386-darwin -S -o - -momit-leaf-frame-pointer %s | \
// RUN:   FileCheck --check-prefix=OMIT_LEAF %s
// OMIT_LEAF: f0:
// OMIT_LEAF-NOT: pushl %ebp
// OMIT_LEAF: ret
// OMIT_LEAF: f1:
// OMIT_LEAF: pushl %ebp
// OMIT_LEAF: ret

void f0() {}
void f1() { f0(); }
