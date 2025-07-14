// REQUIRES: lld

/// Check DTLTO options are forwarded to the linker.

/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang -flto=thin %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 -Xthinlto-distributor=a2,a3 \
// RUN:   -fthinlto-distributor=d.exe -Werror 2>&1 | \
// RUN:   FileCheck %s --check-prefix=FORWARD

// FORWARD: ld.lld
// FORWARD-SAME: "--thinlto-distributor=d.exe"
// FORWARD-SAME: "--thinlto-remote-compiler={{.*}}clang{{[^\"]*}}"
// FORWARD-SAME: "--thinlto-distributor-arg=a1"
// FORWARD-SAME: "--thinlto-distributor-arg=a2"
// FORWARD-SAME: "--thinlto-distributor-arg=a3"

/// Check that options are not added without --thinlto-distributor= and
/// that a warning is issued for unused -Xthinlto-distributor options.
// RUN: %clang -flto=thin %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 -Xthinlto-distributor=a2,a3 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NODIST

// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a1'
// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a2,a3'
// NODIST: ld.lld

/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %clang -flto=thin %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -fthinlto-distributor=d.exe -Werror 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULT

// DEFAULT: ld.lld
// DEFAULT-SAME: "--thinlto-distributor=d.exe"
// DEFAULT-SAME: "--thinlto-remote-compiler={{.*}}clang{{[^\"]*}}"

/// Check that nothing is forwarded when the compiler is not in LTO mode, and that
/// appropriate unused option warnings are issued.
// RUN: %clang %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -fthinlto-distributor=d.exe 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NOFLTO

// NOFLTO: warning: argument unused during compilation: '-fthinlto-distributor=d.exe'
// NOFLTO: ld.lld
