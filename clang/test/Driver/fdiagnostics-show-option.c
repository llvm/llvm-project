/// -fdiagnostics-show-option is the default
// RUN: %clang -### -c %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// ENABLED-NOT: "-fno-diagnostics-show-option"

// RUN: %clang -### -c %s -fdiagnostics-show-option -fno-diagnostics-show-option 2>&1 | \
// RUN:   FileCheck --check-prefix=DISABLED %s
// DISABLED: "-fno-diagnostics-show-option"
