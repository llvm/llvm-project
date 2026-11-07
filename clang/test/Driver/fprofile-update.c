/// For -fprofile-instr-generate and -fprofile-arcs, increment counters atomically
/// if -fprofile-update={atomic,prefer-atomic} is specified.
// RUN: %clang -### %s -c -fprofile-update=atomic 2>&1 | FileCheck %s
// RUN: %clang -### %s -c -fprofile-update=prefer-atomic 2>&1 | FileCheck %s

// CHECK: "-fprofile-update=atomic"

// RUN: %clang -### %s -c -fprofile-update=atomic -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=SINGLE

// SINGLE-NOT: "-fprofile-update=atomic"

// RUN: not %clang %s -c -fprofile-update=unknown 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR: error: unsupported argument 'unknown' to option '-fprofile-update='

/// Under -fsanitize=thread, instrprof counter updates default to atomic in
/// TSan-instrumented functions. An explicit -fprofile-update=single overrides
/// that default with a warning.
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=TSAN-SINGLE
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-generate -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=TSAN-SINGLE
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fcs-profile-generate -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=TSAN-SINGLE
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate -fprofile-update=atomic -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=TSAN-SINGLE
// TSAN-SINGLE: warning: '-fprofile-update=single' with '-fsanitize=thread' may cause ThreadSanitizer to report data races on profile counters [-Wtsan-profile-update]
// TSAN-SINGLE: "-mllvm" "-instrprof-atomic-counter-update-all=false"

/// No warning without -fsanitize=thread
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fprofile-instr-generate -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=NO-WARN

/// No warning with -fsanitize=thread and non-single updates
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate -fprofile-update=atomic 2>&1 | FileCheck %s --check-prefix=NO-WARN
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate -fprofile-update=prefer-atomic 2>&1 | FileCheck %s --check-prefix=NO-WARN
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate -fprofile-update=single -fprofile-update=atomic 2>&1 | FileCheck %s --check-prefix=NO-WARN
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate 2>&1 | FileCheck %s --check-prefix=NO-WARN

/// No warning with -fsanitize=thread when no instrumentation is enabled.
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=NO-WARN
// RUN: %clang -### %s -c --target=x86_64-linux-gnu -fsanitize=thread -fprofile-instr-generate -fno-profile-instr-generate -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=NO-WARN
// NO-WARN-NOT: warning: '-fprofile-update=single'
// NO-WARN-NOT: "-instrprof-atomic-counter-update-all

// AIX specific tests
// RUN: %clang -### %s --target=powerpc-unknown-aix -fprofile-generate -fprofile-update=atomic 2>&1 | FileCheck %s --check-prefix=AIX
// RUN: %clang -### %s --target=powerpc-unknown-aix -fprofile-generate -fprofile-update=prefer-atomic 2>&1 | FileCheck %s --check-prefix=AIX
// RUN: %clang -### %s --target=powerpc-unknown-aix -fprofile-generate 2>&1 | FileCheck %s --check-prefix=AIX-NOATOMIC
// RUN: %clang -### %s --target=powerpc-unknown-aix -fprofile-generate -fprofile-update=single 2>&1 | FileCheck %s --check-prefix=AIX-NOATOMIC
// AIX: "-lcompiler_rt"
// AIX-NOATOMIC-NOT: "-lcompiler_rt"
