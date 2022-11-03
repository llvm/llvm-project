// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// ON-NOT: "-gno-codview-commandline"
// OFF: "-gno-codeview-command-line"

// default
// RUN: %clang_cl /Z7 -### -- %s 2>&1 | FileCheck -check-prefix=ON %s
// enabled
// RUN: %clang_cl /Z7 -gno-codeview-command-line -gcodeview-command-line -### -- %s 2>&1 | FileCheck -check-prefix=ON %s
// disabled
// RUN: %clang_cl /Z7 -gcodeview-command-line -gno-codeview-command-line -### -- %s 2>&1 | FileCheck -check-prefix=OFF %s

// enabled, no /Z7
// RUN: %clang_cl -gcodeview-command-line -### -- %s 2>&1 | FileCheck -check-prefix=ON %s

// GCC-style driver
// RUN: %clang -g -gcodeview -gno-codeview-command-line -gcodeview-command-line -### -- %s 2>&1 | FileCheck -check-prefix=ON %s
// RUN: %clang -g -gcodeview -gcodeview-command-line -gno-codeview-command-line -### -- %s 2>&1 | FileCheck -check-prefix=OFF %s
