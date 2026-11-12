// PR17239 - The /link option, when inside a response file, should only extend
// until the end of the response file (and not the entire command line)

// Note: %s must be preceded by -- or bound to another option, otherwise it may
// be interpreted as a command-line option, e.g. on Mac where %s is commonly
// under /Users.

// RUN: echo /link bar.lib baz.lib > %t.args
// RUN: touch %t.obj
// RUN: %clang_cl -fuse-ld=link -### @%t.args -- %t.obj 2>&1 | FileCheck %s -check-prefix=ARGS
// If the "/link" option captures all remaining args beyond its response file,
// it will also capture "--" and our input argument. In this case, Clang will
// be clueless and will emit "argument unused" warnings. If PR17239 is properly
// fixed, this should not happen because the "/link" option is restricted to
// consume only remaining args in its response file.
// ARGS-NOT: argument unused during compilation
// Identify the linker command
// ARGS: link.exe
