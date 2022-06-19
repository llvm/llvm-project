/// For -fprofile-instr-generate and -fprofile-arcs, increment counters atomically
/// if -fprofile-update={atomic,prefer-atomic} or -fsanitize=thread is specified.
// RUN: %clang -### %s -c -target x86_64-linux -fsanitize=thread %s 2>&1 | FileCheck %s
// RUN: %clang -### %s -c -fprofile-update=atomic 2>&1 | FileCheck %s
// RUN: %clang -### %s -c -fprofile-update=prefer-atomic 2>&1 | FileCheck %s

// CHECK: "-fprofile-update=atomic"

// RUN: %clang -### %s -c -fprofile-update=atomic -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=SINGLE

// SINGLE-NOT: "-fprofile-update=atomic"

// RUN: not %clang %s -c -fprofile-update=unknown 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR: error: unsupported argument 'unknown' to option '-fprofile-update='
