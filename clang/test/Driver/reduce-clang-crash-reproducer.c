// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp %s %t/test.c

// RUN: mkdir -p %t/crash_dump
// RUN: env CLANG_CRASH_DIAGNOSTICS_DIR=%t/crash_dump %clang -target x86_64-unknown-linux-gnu -c %t/test.c || true
// RUN: cp %t/crash_dump/test-*.c %t/
// RUN: cp %t/crash_dump/test-*.sh %t/
// RUN: %python %S/../../utils/reduce-clang-crash.py %t/test-*.sh %t/test-*.c --clang %clang --creduce %S/Inputs/mock-creduce.py -v

// RUN: FileCheck --check-prefix=CHECK-SRC %s < %t/test-*.reduced.c
// RUN: FileCheck --check-prefix=CHECK-CMD %s < %t/test-*.reduced.sh

// CHECK-SRC-NOT: unneeded_function
// CHECK-SRC: #pragma clang __debug crash

// CHECK-CMD: {{^.*}}clang{{(\.exe)?}} -cc1 {{[^ ]*}}test-{{[0-9a-fA-F]+}}.reduced.c{{$}}

void unneeded_function() {}
#pragma clang __debug crash
