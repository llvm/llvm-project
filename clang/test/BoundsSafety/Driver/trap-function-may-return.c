

// RUN: %clang -ftrap-function=foo -ftrap-function-returns -### %s 2>&1 | FileCheck -check-prefix SHOULD-BE-THERE %s
// RUN: %clang -ftrap-function=foo -fno-trap-function-returns -### %s 2>&1 | FileCheck -check-prefix SHOULD %s

// RUN: %clang -ftrap-function=foo -fno-trap-function-returns -ftrap-function-returns -### %s 2>&1 | FileCheck -check-prefix SHOULD-BE-THERE %s
// RUN: %clang -ftrap-function=foo -ftrap-function-returns -fno-trap-function-returns -### %s 2>&1 | FileCheck -check-prefix SHOULD %s

// RUN: not %clang -fsyntax-only -ftrap-function-returns %s 2>&1 > /dev/null
// RUN: %clang -fsyntax-only -fno-trap-function-returns %s 2>&1 > /dev/null

// SHOULD-BE-THERE: -ftrap-function-returns
// SHOULD-NOT: -ftrap-function-return
