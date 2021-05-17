// RUN: touch %t.o
// RUN: %clang -### %t.o -target x86_64-unknown-linux -flto -fpseudo-probe-for-profiling 2>&1 | FileCheck %s --check-prefix=PROBE
// RUN: %clang -### %t.o -target x86_64-unknown-linux -flto=thin -fpseudo-probe-for-profiling 2>&1 | FileCheck %s --check-prefix=PROBE
// RUN: %clang -### %t.o -target x86_64-unknown-linux -flto -fno-pseudo-probe-for-profiling -fpseudo-probe-for-profiling 2>&1 | FileCheck %s --check-prefix=PROBE
// RUN: %clang -### %t.o -target x86_64-unknown-linux -flto 2>&1 | FileCheck %s --check-prefix=NOPROBE
// RUN: %clang -### %t.o -target x86_64-unknown-linux -flto -fno-pseudo-probe-for-profiling 2>&1 | FileCheck %s --check-prefix=NOPROBE
// RUN: %clang -### %t.o -target x86_64-unknown-linux -flto -fpseudo-probe-for-profiling -fno-pseudo-probe-for-profiling 2>&1 | FileCheck %s --check-prefix=NOPROBE

// PROBE: -plugin-opt=pseudo-probe-for-profiling
// NOPROBE-NOT: -plugin-opt=pseudo-probe-for-profiling
