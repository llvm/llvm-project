// RUN: %clang -### -fexperimental-call-graph-section %s 2>&1 | FileCheck --check-prefix=CALL-GRAPH-SECTION %s
// RUN: %clang -### -fexperimental-call-graph-section -fno-experimental-call-graph-section %s 2>&1 | FileCheck --check-prefix=NO-CALL-GRAPH-SECTION %s

// CALL-GRAPH-SECTION: "-fexperimental-call-graph-section"
// NO-CALL-GRAPH-SECTION-NOT: "-fexperimental-call-graph-section"
