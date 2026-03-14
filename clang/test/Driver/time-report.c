// Check that -ftime-report flag is passed to compiler. The value of the flag
// is only diagnosed in the compiler for simplicity since this is a dev option.
// RUN: %clang -### -c -ftime-report %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -ftime-report=per-pass %s 2>&1 | FileCheck %s -check-prefix=PER-PASS
// RUN: %clang -### -c -ftime-report=per-pass-run %s 2>&1 | FileCheck %s -check-prefix=PER-PASS-INVOKE
// RUN: %clang -### -c -ftime-report=unknown %s 2>&1 | FileCheck %s -check-prefix=UNKNOWN

// CHECK:            "-ftime-report"
// PER-PASS:         "-ftime-report=per-pass"
// PER-PASS-INVOKE:  "-ftime-report=per-pass-run"
// UNKNOWN:          "-ftime-report=unknown"
