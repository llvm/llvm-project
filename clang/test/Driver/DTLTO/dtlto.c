// REQUIRES: lld

/// Check DTLTO options are forwarded to the linker.

/// Create a response file for all FileCheck invocations to share. These implicit
/// checks ensure that all lines which mention DTLTO options are checked,
/// and that no unexpected warnings appear.
// RUN: echo " \"%s\" --implicit-check-not=distributor \
// RUN:   --implicit-check-not=remote-compiler \
// RUN:   --implicit-check-not=warning:" > %t_f.rsp

/// Create a response file to check that explicitly specified -Xthinlto-distributor
/// options are forwarded correctly.
// RUN: echo "-flto=thin \"%/s\" -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 \
// RUN:   -Xthinlto-distributor=a2,a3" > %t_l1.rsp

/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang @%t_l1.rsp -fthinlto-distributor=d.exe 2>&1 | \
// RUN:   FileCheck @%t_f.rsp --check-prefix=FORWARD

// FORWARD: ld.lld
// FORWARD-SAME: "--thinlto-distributor=d.exe"
// FORWARD-SAME: "--thinlto-remote-compiler={{.*}}clang{{(.exe)?}}"
// FORWARD-SAME: "--thinlto-distributor-arg=a1"
// FORWARD-SAME: "--thinlto-distributor-arg=a2"
// FORWARD-SAME: "--thinlto-distributor-arg=a3"

/// Check that options are not added without --thinlto-distributor= and
/// that there is an unused option warning issued for -Xthinlto-distributor=
/// options. We specify -flto here as these options should be unaffected by it.
// RUN: %clang @%t_l1.rsp 2>&1 | \
// RUN:   FileCheck @%t_f.rsp --check-prefix=NODIST

// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a1'
// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a2,a3'
// NODIST: ld.lld

/// Create a response file to check the default behavior.
// RUN: echo " \"%/s\" -### -fuse-ld=lld --target=x86_64-linux-gnu \
// RUN:   -fthinlto-distributor=d.exe" > %t_l2.rsp

/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %clang -flto=thin @%t_l2.rsp 2>&1 | \
// RUN:   FileCheck @%t_f.rsp --check-prefix=DEFAULT

// DEFAULT: ld.lld
// DEFAULT-SAME: "--thinlto-distributor=d.exe"
// DEFAULT-SAME: "--thinlto-remote-compiler={{.*}}clang{{(.exe)?}}"
