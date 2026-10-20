// REQUIRES: lld

/// Check DTLTO options are forwarded to the linker.

/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %python %S/Inputs/dtlto-helper.py %clang %llvm-driver > %t_forward.log
// RUN: %clang -flto=thin %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 -Xthinlto-distributor=a2,a3 \
// RUN:   -fthinlto-distributor=d.exe -Werror >>%t_forward.log 2>&1
// RUN: FileCheck %s --input-file=%t_forward.log --check-prefix=FORWARD

// FORWARD:      clang-name:[[CLANG:.*]]
// FORWARD-NEXT: prepend-arg:[[PREPEND_ARG:.*]]
// FORWARD:      ld.lld
// FORWARD-SAME: "--thinlto-distributor=d.exe"
// FORWARD-SAME: "--thinlto-remote-compiler={{[^"]*}}[[CLANG]]"
// FORWARD-SAME: [[PREPEND_ARG]]
// FORWARD-SAME: "--thinlto-distributor-arg=a1"
// FORWARD-SAME: "--thinlto-distributor-arg=a2"
// FORWARD-SAME: "--thinlto-distributor-arg=a3"

/// Check that options are not added without --thinlto-distributor= and
/// that a warning is issued for unused -Xthinlto-distributor options.
// RUN: %clang -flto=thin %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 -Xthinlto-distributor=a2,a3 2>&1 | \
// RUN: FileCheck %s --check-prefix=NODIST --implicit-check-not=distributor \
// RUN:   --implicit-check-not=remote-compiler

// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a1'
// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a2,a3'
// NODIST: ld.lld

/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %python %S/Inputs/dtlto-helper.py %clang %llvm-driver > %t_default.log
// RUN: %clang -flto=thin %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -fthinlto-distributor=d.exe -Werror >>%t_default.log 2>&1
// RUN: FileCheck %s --input-file=%t_default.log --check-prefix=DEFAULT \
// RUN:   --implicit-check-not=distributor --implicit-check-not=remote-compiler

// DEFAULT:      clang-name:[[CLANG:.*]]
// DEFAULT-NEXT: prepend-arg:[[PREPEND_ARG:.*]]
// DEFAULT:      ld.lld
// DEFAULT-SAME: "--thinlto-distributor=d.exe"
// DEFAULT-SAME: "--thinlto-remote-compiler={{[^"]*}}[[CLANG]]"
// DEFAULT-SAME: [[PREPEND_ARG]]

/// Check that nothing is forwarded when the compiler is not in LTO mode, and that
/// appropriate unused option warnings are issued.
// RUN: %clang %s -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -fthinlto-distributor=d.exe 2>&1 | \
// RUN: FileCheck %s --check-prefix=NOFLTO --implicit-check-not=distributor \
// RUN:   --implicit-check-not=remote-compiler

// NOFLTO: warning: argument unused during compilation: '-fthinlto-distributor=d.exe'
// NOFLTO: ld.lld
