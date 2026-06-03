// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp %s %t/test.c

// RUN: echo "%clang -target x86_64-unknown-linux-gnu -c %t/test.c" > %t/crash.sh
// RUN: %python %S/../../utils/reduce-clang-crash.py %t/crash.sh %t/test.c --clang %clang --creduce %S/Inputs/mock-creduce.py -v

// RUN: FileCheck --check-prefix=CHECK-SRC %s < %t/test.reduced.c
// RUN: FileCheck --check-prefix=CHECK-CMD %s < %t/crash.reduced.sh

// CHECK-SRC-NOT: unneeded_function
// CHECK-SRC: #pragma clang __debug crash

// CHECK-CMD: {{^.*}}clang{{(\.exe)?}} -cc1 {{[^ ]*}}test.reduced.c{{$}}

void unneeded_function() {}
#pragma clang __debug crash
