/// This tests uses the PATH environment variable.
// UNSUPPORTED: system-windows, system-cygwin

// RUN: cd %S

/// If --ld-path= specifies a word (without /), -B and COMPILER_PATH are
/// consulted to locate the linker.
// RUN: %clang %s -### -B %S/Inputs/basic_freebsd_tree/usr/bin --ld-path=ld.bfd \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BFD
// RUN: env COMPILER_PATH=%S/Inputs/basic_freebsd_tree/usr/bin %clang %s -### --ld-path=ld.bfd \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BFD
/// Then PATH is consulted.
// RUN: env PATH=%S/Inputs/basic_freebsd_tree/usr/bin %clang %s -### --ld-path=ld.bfd \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BFD

// BFD: Inputs/basic_freebsd_tree/usr/bin/ld.bfd"

// RUN: env PATH=%S/Inputs/basic_freebsd_tree/usr/bin %clang %s -### --ld-path=ld.gold \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=GOLD

// GOLD: Inputs/basic_freebsd_tree/usr/bin/ld.gold"

// RUN: not env COMPILER_PATH= PATH=%S/Inputs/basic_freebsd_tree/usr/bin %clang %s -### --ld-path=not_exist \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NOT_EXIST

// NOT_EXIST: error: invalid linker name in argument '--ld-path=not_exist'

// RUN: not %clang %s -### --ld-path= \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=EMPTY

// EMPTY: error: invalid linker name in argument '--ld-path='

/// If --ld-path= contains a slash, PATH is not consulted.
// RUN: env COMPILER_PATH=%S/Inputs/basic_freebsd_tree/usr/bin not %clang %s -### --ld-path=./ld.bfd \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NO_BFD

// NO_BFD: error: invalid linker name in argument '--ld-path=./ld.bfd'

/// --ld-path can specify an absolute path.
// RUN: %clang %s -### --ld-path=%S/Inputs/basic_freebsd_tree/usr/bin/ld.bfd \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BFD

// RUN: %clang %s -### --ld-path=Inputs/basic_freebsd_tree/usr/bin/ld.bfd \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BFD

/// --ld-path= and -fuse-ld= can be used together. --ld-path= takes precedence.
/// -fuse-ld= can be used to specify the linker flavor.
// RUN: %clang %s -### -Werror --ld-path=%S/Inputs/basic_freebsd_tree/usr/bin/ld.bfd -fuse-ld=gold \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BFD --implicit-check-not=error:

/// --ld-path= respects -working-directory.
// RUN: %clang %s -### --ld-path=usr/bin/ld.bfd -working-directory=%S/Inputs/basic_freebsd_tree \
// RUN:   --target=x86_64-unknown-freebsd --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 | \
// RUN:   FileCheck %s --check-prefix=USR_BIN_BFD

// USR_BIN_BFD: "usr/bin/ld.bfd"
