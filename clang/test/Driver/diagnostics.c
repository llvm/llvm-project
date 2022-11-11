// Parse diagnostic arguments in the driver

// Exactly which arguments are warned about and which aren't differ based
// on what target is selected. -stdlib= and -fuse-ld= emit diagnostics when
// compiling C code, for e.g. *-linux-gnu. Linker inputs, like -lfoo, emit
// diagnostics when only compiling for all targets.

// This is normally a non-fatal warning:
// RUN: %clang --target=x86_64-apple-darwin11 \
// RUN:   -fsyntax-only -lfoo %s 2>&1 | FileCheck %s

// Either with a specific -Werror=unused.. or a blanket -Werror, this
// causes the command to fail.
// RUN: not %clang --target=x86_64-apple-darwin11 \
// RUN:   -fsyntax-only -lfoo \
// RUN:   -Werror=unused-command-line-argument %s 2>&1 | FileCheck %s

// RUN: not %clang --target=x86_64-apple-darwin11 \
// RUN:   -fsyntax-only -lfoo -Werror %s 2>&1 | FileCheck %s

// With a specific -Wno-..., no diagnostic should be printed.
// RUN: %clang --target=x86_64-apple-darwin11 \
// RUN:   -fsyntax-only -lfoo -Werror \
// RUN:   -Wno-unused-command-line-argument %s 2>&1 | count 0

// With -Qunused-arguments, no diagnostic should be printed.
// RUN: %clang --target=x86_64-apple-darwin11 \
// RUN:   -fsyntax-only -lfoo -Werror \
// RUN:   -Qunused-arguments %s 2>&1 | count 0

// With the argument enclosed in --{start,end}-no-unused-arguments,
// there's no diagnostic.
// RUN: %clang --target=x86_64-apple-darwin11 -fsyntax-only \
// RUN:   --start-no-unused-arguments -lfoo --end-no-unused-arguments \
// RUN:   -Werror %s 2>&1 | count 0

// With --{start,end}-no-unused-argument around a different argument, it
// still warns about the unused argument.
// RUN: not %clang --target=x86_64-apple-darwin11 \
// RUN:   --start-no-unused-arguments -fsyntax-only --end-no-unused-arguments \
// RUN:   -lfoo -Werror %s 2>&1 | FileCheck %s

// Test clang-cl warning about unused linker options.
// RUN: not %clang_cl -fsyntax-only /WX \
// RUN:   -LD -- %s 2>&1 | FileCheck %s --check-prefix=CL-WARNING

// Test clang-cl ignoring the warning with --start-no-unused-arguments.
// RUN: %clang_cl -fsyntax-only /WX \
// RUN:   --start-no-unused-arguments /LD --end-no-unused-arguments -- %s 2>&1 | count 0

// CHECK: -lfoo: 'linker' input unused

// CL-WARNING: argument unused during compilation: '-LD'
