// REQUIRES: backtrace
// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t && mkdir %t
// RUN: cd %t
// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=repro.tar -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=TAR
// RUN: mkdir extract
// RUN: tar -xf repro.tar -C extract
// RUN: FileCheck %s --check-prefix=SH < extract/*/crash-diagnostics-tar-*.sh
// RUN: FileCheck %s --check-prefix=C < extract/*/crash-diagnostics-tar-*.c

// RUN: not %crash_opt %clang -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOTAR

// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=%t/nonexistent/repro.tar -c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID

// RUN: not %crash_opt %clang -fcrash-diagnostics-tar=repro-stdin.tar -c -x c - -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=STDIN
// RUN: not ls repro-stdin.tar

#pragma clang __debug parser_crash

// TAR: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
// TAR: repro.tar
// TAR-NOT: .c{{$}}
// TAR-NOT: .sh{{$}}

// NOTAR: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
// NOTAR: .c{{$}}
// NOTAR: .sh{{$}}
// NOTAR-NOT: .tar{{$}}

// INVALID: Error creating reproducer tarball:

// SH: # Crash reproducer for
// C: # 1 "

// STDIN: PLEASE submit a bug report to
// STDIN: note: diagnostic msg: Error generating preprocessed source(s) - ignoring input from stdin.
// STDIN: note: diagnostic msg: Error generating preprocessed source(s) - no preprocessable inputs.
