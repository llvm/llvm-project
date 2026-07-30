// RUN: env NO_COLOR=1 %clang -### %s 2>&1 | FileCheck --check-prefix=NO-COLOR %s
// RUN: env NO_COLOR=1 %clang -fcolor-diagnostics -### %s 2>&1 | FileCheck --check-prefix=COLOR %s
// RUN: env NO_COLOR=1 %clang -fdiagnostics-color=auto -### %s 2>&1 | FileCheck --check-prefix=NO-COLOR %s
// RUN: env NO_COLOR=1 %clang -fdiagnostics-color=always -### %s 2>&1 | FileCheck --check-prefix=COLOR %s
// RUN: env NO_COLOR=1 %clang -fdiagnostics-color=never -### %s 2>&1 | FileCheck --check-prefix=NO-COLOR %s

// Note, the value of the environment variable does not matter, only that it is defined and not empty.
// RUN: env NO_COLOR=0 %clang -### %s 2>&1 | FileCheck --check-prefix=NO-COLOR %s
// Note, an empty value means we automatically decide whether to enable colors or not, and lit tests
// are not run in a PTY, so colors are disabled by default. There is no easy way for us to test this
// configuration where auto enables colors.
// RUN: env NO_COLOR= %clang -### %s 2>&1 | FileCheck --check-prefix=NO-COLOR %s

int main(void) {}

// COLOR: -fcolor-diagnostics
// NO-COLOR-NOT: -fcolor-diagnostics
