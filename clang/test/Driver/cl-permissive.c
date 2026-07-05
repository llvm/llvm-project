// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /permissive -### -- %s 2>&1 | FileCheck -check-prefix=PERMISSIVE %s
// PERMISSIVE: "-fno-operator-names"
// PERMISSIVE: "-fdelayed-template-parsing"
// RUN: %clang_cl /permissive- -### -- %s 2>&1 | FileCheck -check-prefix=PERMISSIVE-MINUS %s
// PERMISSIVE-MINUS-NOT: "-fno-operator-names"
// PERMISSIVE-MINUS-NOT: "-fdelayed-template-parsing"

// The switches set by permissive may then still be manually enabled or disabled
// RUN: %clang_cl /permissive /Zc:twoPhase -### -- %s 2>&1 | FileCheck -check-prefix=PERMISSIVE-OVERWRITE %s
// PERMISSIVE-OVERWRITE: "-fno-operator-names"
// PERMISSIVE-OVERWRITE-NOT: "-fdelayed-template-parsing"
// RUN: %clang_cl /permissive- /Zc:twoPhase- -### -- %s 2>&1 | FileCheck -check-prefix=PERMISSIVE-MINUS-OVERWRITE %s
// PERMISSIVE-MINUS-OVERWRITE-NOT: "-fno-operator-names"
// PERMISSIVE-MINUS-OVERWRITE: "-fdelayed-template-parsing"
