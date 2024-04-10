// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.
// REQUIRES: coff-supported-target

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

// RUN: %clang_cl /Zi /c -### -- %s 2>&1 | FileCheck -check-prefix=Zi %s
// Zi: "-gcodeview"
// Zi: "-debug-info-kind=constructor"

// RUN: %clang_cl /Z7 /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7 %s
// Z7: "-gcodeview"
// Z7: "-debug-info-kind=constructor"

// RUN: %clang_cl -gline-tables-only /c -### -- %s 2>&1 | FileCheck -check-prefix=ZGMLT %s
// ZGMLT: "-gcodeview"
// ZGMLT: "-debug-info-kind=line-tables-only"

// RUN: %clang_cl /Z7 /Foa.obj -### -- %s 2>&1 | FileCheck -check-prefix=ABSOLUTE_OBJPATH %s
// ABSOLUTE_OBJPATH: "-object-file-name={{.*}}a.obj"

// RUN: %clang_cl -fdebug-compilation-dir=. /Z7 /Foa.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH1 %s
// RELATIVE_OBJPATH1: "-object-file-name=a.obj"

// RUN: %clang_cl -fdebug-compilation-dir=. /Z7 /Fo:a.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH1_COLON %s
// RELATIVE_OBJPATH1_COLON: "-object-file-name=a.obj"

// RUN: %clang_cl -fdebug-compilation-dir=. /Z7 /Fofoo/a.obj -### -- %s 2>&1 | FileCheck -check-prefix=RELATIVE_OBJPATH2 %s
// RELATIVE_OBJPATH2: "-object-file-name=foo\\a.obj"
