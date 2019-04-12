// For these next two tests when optimized we should omit the leaf frame
// pointer, for unoptimized we should have a leaf frame pointer.
// RUN: %clang -### -target i386-pc-linux-gnu -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=LINUX-OPT %s
// LINUX-OPT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target i386-pc-linux-gnu -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=LINUX %s
// LINUX-NOT: "-momit-leaf-frame-pointer"

// CloudABI follows the same rules as Linux.
// RUN: %clang -### -target x86_64-unknown-cloudabi -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CLOUDABI-OPT %s
// CLOUDABI-OPT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target x86_64-unknown-cloudabi -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CLOUDABI %s
// CLOUDABI-NOT: "-momit-leaf-frame-pointer"

// NetBSD follows the same rules as Linux.
// RUN: %clang -### -target x86_64-unknown-netbsd -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NETBSD-OPT %s
// NETBSD-OPT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target x86_64-unknown-netbsd -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=NETBSD %s
// NETBSD-NOT: "-momit-leaf-frame-pointer"

// OpenBSD follows the same rules as Linux.
// RUN: %clang -### -target x86_64-unknown-openbsd -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OPENBSD-OPT %s
// OPENBSD-OPT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target x86_64-unknown-openbsd -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OPENBSD %s
// OPENBSD-NOT: "-momit-leaf-frame-pointer"

// Darwin disables omitting the leaf frame pointer even under optimization
// unless the command lines are given.
// RUN: %clang -### -target i386-apple-darwin -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DARWIN %s
// DARWIN: "-mdisable-fp-elim"

// RUN: %clang -### -target i386-apple-darwin -S -O1 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=DARWIN-OPT %s
// DARWIN-OPT-NOT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target i386-darwin -S -fomit-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OMIT_ALL %s
// OMIT_ALL-NOT: "-mdisable-fp-elim"

// RUN: %clang -### -target i386-darwin -S -momit-leaf-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OMIT_LEAF %s
// OMIT_LEAF: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target armv7s-apple-ios -fomit-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=WARN-OMIT-7S %s
// WARN-OMIT-7S: warning: optimization flag '-fomit-frame-pointer' is not supported for target 'armv7s'
// WARN-OMIT-7S: "-mdisable-fp-elim"
// WARN-OMIT-7S-NOT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target armv7k-apple-watchos -fomit-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=WARN-OMIT-7K %s
// WARN-OMIT-7K: warning: optimization flag '-fomit-frame-pointer' is not supported for target 'armv7k'
// WARN-OMIT-7K: "-mdisable-fp-elim"
// WARN-OMIT-7K-NOT: "-momit-leaf-frame-pointer"

// RUN: %clang -### -target armv7s-apple-ios8.0 -momit-leaf-frame-pointer %s 2>&1 | \
// RUN:   FileCheck --check-prefix=WARN-OMIT-LEAF-7S %s
// WARN-OMIT-LEAF-7S-NOT: warning: optimization flag '-momit-leaf-frame-pointer' is not supported for target 'armv7s'
// WARN-OMIT-LEAF-7S: "-mdisable-fp-elim"
// WARN-OMIT-LEAF-7S: "-momit-leaf-frame-pointer"

// On the PS4, we default to omitting the frame pointer on leaf functions
// (OMIT_LEAF check line is above)
// RUN: %clang -### -target x86_64-scei-ps4 -S %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OMIT_LEAF %s
// RUN: %clang -### -target x86_64-scei-ps4 -S -O2 %s 2>&1 | \
// RUN:   FileCheck --check-prefix=OMIT_LEAF %s

void f0() {}
void f1() { f0(); }
