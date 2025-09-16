// REQUIRES: lld

/// https://github.com/llvm/llvm-project/issues/159125.
// XFAIL: llvm-driver

/// Check DTLTO options are forwarded to the linker.

/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang -flto=thin %s -### --target=x86_64-sie-ps5 \
// RUN:   -Xthinlto-distributor=a1 -Xthinlto-distributor=a2,a3 \
// RUN:   -fthinlto-distributor=d.exe -Werror 2>&1 | \
// RUN:   FileCheck %s --check-prefix=FORWARD

// FORWARD: prospero-lld
// FORWARD-SAME: "--thinlto-distributor=d.exe"
// FORWARD-SAME: "--thinlto-remote-compiler={{.*}}clang{{[^\"]*}}"
// FORWARD-SAME: "--thinlto-distributor-arg=a1"
// FORWARD-SAME: "--thinlto-distributor-arg=a2"
// FORWARD-SAME: "--thinlto-distributor-arg=a3"

/// Check that options are not added without --thinlto-distributor= and
/// that a warning is issued for unused -Xthinlto-distributor options.
// RUN: %clang -flto=thin %s -### --target=x86_64-sie-ps5 \
// RUN:   -Xthinlto-distributor=a1 -Xthinlto-distributor=a2,a3 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NODIST --implicit-check-not=distributor \
// RUN:     --implicit-check-not=remote-compiler

// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a1'
// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a2,a3'
// NODIST: prospero-lld

/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %clang -flto=thin %s -### --target=x86_64-sie-ps5 \
// RUN:   -fthinlto-distributor=d.exe -Werror 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULT --implicit-check-not=distributor \
// RUN:     --implicit-check-not=remote-compiler

// DEFAULT: prospero-lld
// DEFAULT-SAME: "--thinlto-distributor=d.exe"
// DEFAULT-SAME: "--thinlto-remote-compiler={{[^"]+}}"

/// Check that the arguments are forwarded unconditionally even when the
/// compiler is not in LTO mode.
// RUN: %clang %s -### --target=x86_64-sie-ps5 \
// RUN:   -fthinlto-distributor=d.exe -Werror 2>&1 | \
// RUN:   FileCheck %s --check-prefix=DEFAULT --implicit-check-not=distributor \
// RUN:     --implicit-check-not=remote-compiler
