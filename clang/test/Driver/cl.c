// Note: we have to quote the /? option, otherwise some shells will try to
// expand the ? into a one-letter filename in the root directory, and make
// the test fail is such a file or directory exists.

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// Check that clang-cl options are not available by default.
// RUN: %clang -help | FileCheck %s -check-prefix=DEFAULT
// DEFAULT-NOT: CL.EXE COMPATIBILITY OPTIONS
// DEFAULT-NOT: {{/[?]}}
// DEFAULT-NOT: /help
// RUN: not %clang "/?"
// RUN: not %clang -?
// RUN: not %clang /help

// Check that /? and /help are available as clang-cl options.
// RUN: %clang_cl "/?" | FileCheck %s -check-prefix=CL
// RUN: %clang_cl /help | FileCheck %s -check-prefix=CL
// RUN: %clang_cl -help | FileCheck %s -check-prefix=CL
// CL: CL.EXE COMPATIBILITY OPTIONS
// CL: {{/[?]}}
// CL: /help

// Options which are not "core" clang options nor cl.exe compatible options
// are not available in clang-cl.
// DEFAULT: -fapple-kext
// CL-NOT: -fapple-kext

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=COMPILE %s
// COMPILE: "-fdiagnostics-format" "msvc"
