// -fbacktrace is a Flang option, but it stays visible to Clang so a build
// that passes one flag set to both drivers does not fail. Clang does not
// forward it to cc1.
//
// RUN: %clang -### -fbacktrace -c -x c %s 2>&1 | FileCheck %s --check-prefix=YES
// RUN: %clang -### -fno-backtrace -c -x c %s 2>&1 | FileCheck %s --check-prefix=NO

// YES: warning: argument unused during compilation: '-fbacktrace'
// YES-NOT: error:
// NO: warning: argument unused during compilation: '-fno-backtrace'
// NO-NOT: error:

int flang_backtrace_anchor(void) { return 0; }
